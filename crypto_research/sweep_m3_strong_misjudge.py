from __future__ import annotations
from itertools import product
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import acquire_klines, acquire_funding

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["5m", "15m", "30m", "1h"]
TF_RULE = {"5m":"5min", "15m":"15min", "30m":"30min", "1h":"1h"}
PIP_FX = 0.0001  # FX-specific; deliberately NOT carried into crypto.
FEE_BPS = 5.0
SLIPPAGE_BPS = 2.0

def wilder(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ix = np.flatnonzero(np.isfinite(x))
    if len(ix) < n: return out
    first = ix[0]; acc = x[first]; count = 1
    for i in range(first + 1, len(x)):
        if not np.isfinite(x[i]): continue
        acc = (1 - 1/n) * acc + x[i] / n
        count += 1
        if count >= n: out[i] = acc
    return out

def prepare(raw, timeframe, atr_n=20, adx_n=14, rsi_n=14):
    rule = TF_RULE[timeframe]
    d = raw.resample(rule, label="right", closed="left").agg({
        "open":"first","high":"max","low":"min","close":"last",
        "volume":"sum","quote_volume":"sum","trades":"sum",
        "taker_buy_base":"sum","taker_buy_quote":"sum"
    }).dropna(subset=["open","high","low","close"]).copy()
    o,h,l,c = [d[k].to_numpy(float) for k in ("open","high","low","close")]
    prev = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h-l, np.abs(h-prev), np.abs(l-prev)])
    atr = np.r_[np.nan, wilder(tr, atr_n)[:-1]]  # prior completed ATR, exact Strong Misjudge timing
    delta = np.r_[np.nan, np.diff(c)]
    ag = wilder(np.maximum(delta, 0), rsi_n)
    al = wilder(np.maximum(-delta, 0), rsi_n)
    rs = np.divide(ag, al, out=np.full_like(ag, np.nan), where=al != 0)
    rsi = 100 - 100/(1+rs)
    up = np.r_[np.nan, np.diff(h)]; down = -np.r_[np.nan, np.diff(l)]
    pdm = np.where((up>down)&(up>0), up, 0.0)
    mdm = np.where((down>up)&(down>0), down, 0.0)
    a = wilder(tr, adx_n)
    pdi = 100*wilder(pdm, adx_n)/a
    mdi = 100*wilder(mdm, adx_n)/a
    dx = np.divide(100*np.abs(pdi-mdi), pdi+mdi, out=np.full_like(pdi,np.nan), where=(pdi+mdi)!=0)
    adx = wilder(dx, adx_n)
    d["atr_prior"] = atr; d["rsi"] = rsi; d["adx"] = adx
    return d

def make_signals(d, atr_mult=2.5, adx_lo=18, adx_hi=30, rsi_mode="exact", retrace=0.82):
    x=d.copy()
    o,h,l,c=[x[k].to_numpy(float) for k in ("open","high","low","close")]
    rng=h-l
    direction=np.where(c>o,1,np.where(c<o,-1,0))
    atr=x.atr_prior.to_numpy(float); adx=x.adx.to_numpy(float); rsi=x.rsi.to_numpy(float)
    adx_ok=(adx>=adx_lo)&(adx<=adx_hi)
    if rsi_mode=="exact":
        rsi_ok=((direction==1)&(rsi>=50)&(rsi<=70))|((direction==-1)&(rsi>=30)&(rsi<=50))
    elif rsi_mode=="loose":
        rsi_ok=((direction==1)&(rsi>=50)&(rsi<=75))|((direction==-1)&(rsi>=25)&(rsi<=50))
    elif rsi_mode=="none":
        rsi_ok=np.ones(len(x),dtype=bool)
    else: raise ValueError(rsi_mode)
    valid=np.isfinite(atr)&np.isfinite(adx)&np.isfinite(rsi)&(atr>0)&(rng>0)
    x["signal"]=np.where(valid&(rng>=atr_mult*atr)&adx_ok&rsi_ok,direction,0).astype(int)
    x["entry_level"]=np.where(x.signal.to_numpy()==1,h-retrace*rng,np.where(x.signal.to_numpy()==-1,l+retrace*rng,np.nan))
    x["stop_level"]=np.where(x.signal.to_numpy()==1,l,np.where(x.signal.to_numpy()==-1,h,np.nan))
    return x

def backtest(d, funding, symbol, model, timeframe, rr=4.0, pending_bars=20, fee_bps=FEE_BPS, slip_bps=SLIPPAGE_BPS, trade_start=None):
    trades=[]; pending=None; pos=None; last_submitted_signal_i=-1
    max_hold={"5m":288,"15m":96,"30m":48,"1h":24}[timeframe]
    ix=d.index
    def settle(exit_px, ts, reason):
        side=pos["side"]
        exit_fill=exit_px*(1-side*slip_bps/10000)
        gross=side*(exit_fill-pos["entry"])/pos["entry"]
        fund_cost=0.0
        if funding is not None and not funding.empty:
            ff=funding.loc[(funding.index>=pos["entry_time"])&(funding.index<=ts)]
            fund_cost=sum(side*float(v) for v in ff.funding_rate)
        net=gross-2*fee_bps/10000-fund_cost
        risk=pos["risk"]/pos["entry"]
        trades.append({"symbol":symbol,"model":model,"timeframe":timeframe,"side":"LONG" if side==1 else "SHORT",
          "entry_time":str(pos["entry_time"]),"exit_time":str(ts),"entry":pos["entry"],"exit":exit_fill,
          "initial_stop":pos["stop"],"target":pos["target"],"exit_reason":reason,
          "gross_bps":gross*10000,"fees_bps":2*fee_bps,"funding_cost_bps":fund_cost*10000,
          "net_bps":net*10000,"net_R":net/risk if risk else 0.0})
    for i in range(1,len(d)):
        row=d.iloc[i]; ts=ix[i]; o=float(row.open); h=float(row.high); l=float(row.low); c=float(row.close)
        # Manage an already-open position first; stop wins ambiguous same-bar collisions.
        if pos is not None:
            held=i-pos["entry_i"]
            stop_hit=(l<=pos["stop"]) if pos["side"]==1 else (h>=pos["stop"])
            target_hit=(h>=pos["target"]) if pos["side"]==1 else (l<=pos["target"])
            if stop_hit or target_hit or held>=max_hold:
                if stop_hit:
                    px=o if ((pos["side"]==1 and o<pos["stop"]) or (pos["side"]==-1 and o>pos["stop"])) else pos["stop"]
                    reason="STOP"
                elif target_hit: px=pos["target"]; reason="TARGET"
                else: px=c; reason="TIME"
                settle(float(px),ts,reason); pos=None
        # Submit the just-closed candle's limit order before evaluating this bar,
        # so the order is active from the first tradable bar after the signal.
        signal_i=i-1
        if pos is None and pending is None and signal_i>last_submitted_signal_i and (trade_start is None or ts >= trade_start) and int(d.signal.iloc[signal_i])!=0:
            side=int(d.signal.iloc[signal_i])
            pending={"side":side,"entry":float(d.entry_level.iloc[signal_i]),"stop":float(d.stop_level.iloc[signal_i]),"signal_i":signal_i}
            last_submitted_signal_i=signal_i
        # Pending limit: 20 bars from the first bar after signal.
        if pos is None and pending is not None:
            age=i-pending["signal_i"]
            if age>pending_bars: pending=None
            else:
                side=pending["side"]; level=pending["entry"]; stop=pending["stop"]
                touched=(l<=level) if side==1 else (h>=level)
                gap_invalid=(o<=stop) if side==1 else (o>=stop)
                if touched and not gap_invalid:
                    raw_fill=min(o,level) if side==1 else max(o,level)
                    entry=raw_fill*(1+side*slip_bps/10000)
                    risk=abs(entry-stop)
                    if risk>0 and ((side==1 and stop<entry) or (side==-1 and stop>entry)):
                        pos={"side":side,"entry":entry,"stop":stop,"risk":risk,
                             "target":entry+side*rr*risk,"entry_time":ts,"entry_i":i}
                        # Conservative same-candle handling after a limit fill.
                        sh=(l<=stop) if side==1 else (h>=stop)
                        th=(h>=pos["target"]) if side==1 else (l<=pos["target"])
                        if sh or th:
                            if sh:
                                px=o if ((side==1 and o<stop) or (side==-1 and o>stop)) else stop
                                reason="STOP"
                            else: px=pos["target"]; reason="TARGET"
                            settle(float(px),ts,reason); pos=None
                    pending=None
    if pos is not None: settle(float(d.close.iloc[-1]),ix[-1],"END_OF_SAMPLE")
    return pd.DataFrame(trades)

def metrics(t):
    if t.empty: return {"trades":0,"win_rate_pct":0.0,"profit_factor":0.0,"net_bps":0.0,"max_drawdown_pct":0.0}
    t=t.sort_values("exit_time"); wins=t[t.net_bps>0]; losses=t[t.net_bps<=0]
    gp=float(wins.net_bps.sum()); gl=abs(float(losses.net_bps.sum()))
    eq=(1+t.net_bps/10000).cumprod(); dd=(eq/eq.cummax()-1).min()*100
    return {"trades":len(t),"win_rate_pct":round(100*len(wins)/len(t),2),
      "profit_factor":round(gp/gl,4) if gl else None,"net_bps":round(float(t.net_bps.sum()),2),
      "max_drawdown_pct":round(float(dd),3),"avg_net_bps":round(float(t.net_bps.mean()),3)}

def main():
    out=Path("artifacts"); out.mkdir(exist_ok=True)
    raw={}; funds={}
    for s in SYMBOLS:
        raw[s]=acquire_klines(s,"2024-01-01","2026-01-01")
        funds[s]=acquire_funding(s,"2024-01-01","2026-01-01")
    # Exact rule replication first, with 82% limit entry and 4R target.
    exact_rows=[]; exact_trades=[]
    for tf in TIMEFRAMES:
      for s in SYMBOLS:
        d=make_signals(prepare(raw[s],tf,20),2.5,18,30,"exact",.82)
        cut=pd.Timestamp("2025-01-01",tz="UTC")
        train_d=d.loc[d.index<cut]; test_f=funds[s].loc[funds[s].index<cut]
        train_t=backtest(train_d,test_f,s,"SM_EXACT_2.5ATR",tf,4.0)
        test_t=backtest(d,funds[s],s,"SM_EXACT_2.5ATR",tf,4.0,trade_start=cut)
        for sample,t in [("TRAIN_2024",train_t),("OOS_2025",test_t)]:
          if not t.empty:
            exact_trades.append(t)
            exact_rows.append({"model":"SM_EXACT_2.5ATR","timeframe":tf,"symbol":s,"sample":sample,**metrics(t)})
    # One-factor and interaction grid on 15m. Select only on 2024; 2025 remains frozen.
    grid=[]
    configs=list(product([14,20,25],[2.2,2.4,2.5,2.7],[(18,30),(15,25),(20,35)],
                         ["exact","loose","none"],[.75,.786,.80,.82,.85]))
    prepared={s:{n:prepare(raw[s],"15m",n) for n in [14,20,25]} for s in SYMBOLS}
    for cid,(atr_n,mult,adx_band,rsi_mode,retrace) in enumerate(configs,1):
      rr=4.0
      pieces=[]
      for s in SYMBOLS:
        d=make_signals(prepared[s][atr_n],mult,adx_band[0],adx_band[1],rsi_mode,retrace)
        cut=pd.Timestamp("2025-01-01",tz="UTC")
        train_d=d.loc[d.index<cut]; train_f=funds[s].loc[funds[s].index<cut]
        t=backtest(train_d,train_f,s,f"SMG{cid}","15m",rr)
        if not t.empty: pieces.append(t)
      if not pieces: continue
      all_t=pd.concat(pieces,ignore_index=True); sm=metrics(all_t)
      # Avoid tiny samples masquerading as high PF.
      if sm["trades"]>=60:
        grid.append({"config_id":cid,"atr_period":atr_n,"impulse_atr":mult,"adx_min":adx_band[0],
          "adx_max":adx_band[1],"rsi_mode":rsi_mode,"retracement":retrace,"rr":rr,**sm})
    g=pd.DataFrame(grid)
    if not g.empty:
      g=g.sort_values(["profit_factor","net_bps"],ascending=False)
      g.to_csv(out/"strong_misjudge_crypto_grid_2024.csv",index=False)
      top=g.head(20)
      oos=[]
      # Recompute selected 2024 leaders, then evaluate the frozen 2025 data.
      rr_train=[]
      for _,r in top.iterrows():
        atr_n=int(r.atr_period)
        for rr_test in [2.0,3.0,4.0,5.0]:
          train_parts=[]
          for s in SYMBOLS:
            d=make_signals(prepared[s][atr_n],float(r.impulse_atr),int(r.adx_min),int(r.adx_max),str(r.rsi_mode),float(r.retracement))
            cut=pd.Timestamp("2025-01-01",tz="UTC")
            td=d.loc[d.index<cut]; tfund=funds[s].loc[funds[s].index<cut]
            tr=backtest(td,tfund,s,f"SMG{int(r.config_id)}","15m",rr_test)
            if not tr.empty: train_parts.append(tr)
            te=backtest(d,funds[s],s,f"SMG{int(r.config_id)}","15m",rr_test,trade_start=cut)
            if not te.empty: oos.append({"config_id":int(r.config_id),"symbol":s,"atr_period":atr_n,
              "impulse_atr":r.impulse_atr,"adx_min":r.adx_min,"adx_max":r.adx_max,"rsi_mode":r.rsi_mode,
              "retracement":r.retracement,"rr":rr_test,**metrics(te)})
          if train_parts:
            rr_train.append({"config_id":int(r.config_id),"atr_period":atr_n,"impulse_atr":r.impulse_atr,
              "adx_min":r.adx_min,"adx_max":r.adx_max,"rsi_mode":r.rsi_mode,
              "retracement":r.retracement,"rr":rr_test,**metrics(pd.concat(train_parts,ignore_index=True))})
      pd.DataFrame(rr_train).to_csv(out/"strong_misjudge_rr_train_2024.csv",index=False)
      pd.DataFrame(oos).to_csv(out/"strong_misjudge_crypto_oos_2025.csv",index=False)
    pd.DataFrame(exact_rows).to_csv(out/"strong_misjudge_exact_timeframe_comparison.csv",index=False)
    if exact_trades: pd.concat(exact_trades,ignore_index=True).to_csv(out/"strong_misjudge_exact_trades.csv",index=False)
    print("EXACT TIMEFRAME COMPARISON"); print(pd.DataFrame(exact_rows).to_string(index=False))
    print("15M 2024 GRID TOP 20"); print(g.head(20).to_string(index=False) if not g.empty else "No eligible configurations")
    print("2025 FROZEN OOS"); print(pd.read_csv(out/"strong_misjudge_crypto_oos_2025.csv").to_string(index=False) if (out/"strong_misjudge_crypto_oos_2025.csv").exists() else "No OOS candidates")
if __name__=="__main__": main()
