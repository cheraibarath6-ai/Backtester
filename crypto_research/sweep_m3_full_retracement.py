from __future__ import annotations
"""Model 3 multi-candle limit-retracement research.

A completed impulse candle arms one exact-price limit order. The order may fill
on any of the next 20 bars. No retracement-candle or confirmation-candle rule.
One pending order / one position per symbol and timeframe; stop-first ambiguity.
"""
from itertools import product
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from .data import acquire_klines, acquire_funding, quality_report
from .sweep_m3_strong_misjudge import prepare, wilder

SYMBOLS=("BTCUSDT","ETHUSDT")
TIMEFRAMES=("5m","15m","30m","1h")
RETRACEMENTS=(.70,.725,.75,.77,.786,.80,.81,.82,.83,.85,.875,.90)
ATR_PERIODS=(20,)
IMPULSE_ATR=(2.4,2.5)
ADX_BANDS=((18,30),)
RSI_MODES=("exact","none")
RRS=(2.,3.,4.)
PENDING_BARS=(3,6,10)
START="2021-01-01"; END="2026-11-01"
CUTS=(("TRAIN_2021_2023","2021-01-01","2024-01-01"),
      ("OOS_2024","2024-01-01","2025-01-01"),
      ("OOS_2025","2025-01-01","2026-01-01"),
      ("OOS_2026_YTD","2026-01-01","2026-11-01"))
FEE=5.; SLIP=2.

def make_impulses(d, mult, adxlo, adxhi, mode):
    o,h,l,c=(d[k].to_numpy(float) for k in ("open","high","low","close"))
    rng=h-l; direction=np.where(c>o,1,np.where(c<o,-1,0))
    atr=d.atr_prior.to_numpy(float); adx=d.adx.to_numpy(float); rsi=d.rsi.to_numpy(float)
    adxok=(adx>=adxlo)&(adx<=adxhi)
    if mode=="exact":
        rsok=((direction==1)&(rsi>=50)&(rsi<=70))|((direction==-1)&(rsi>=30)&(rsi<=50))
    elif mode=="loose":
        rsok=((direction==1)&(rsi>=50)&(rsi<=75))|((direction==-1)&(rsi>=25)&(rsi<=50))
    else: rsok=np.ones(len(d),bool)
    valid=np.isfinite(atr)&np.isfinite(adx)&(atr>0)&(rng>0)
    if mode!="none": valid &= np.isfinite(rsi)
    sig=np.where(valid&(rng>=mult*atr)&adxok&rsok,direction,0).astype(np.int8)
    return sig

def run_limit_bt(d, funding, sig, retrace, rr, symbol, tf, model, sample_start, sample_end, pending_bars):
    # Iterate from signal close; limit order becomes live on following candle.
    ix=d.index; o=d.open.to_numpy(float); h=d.high.to_numpy(float); l=d.low.to_numpy(float); c=d.close.to_numpy(float)
    side=0; entry=stop=target=risk=0.; entry_i=-1; entry_ts=None
    pending=None; trades=[]
    hold={"5m":288,"15m":96,"30m":48,"1h":24}[tf]
    fs=funding.funding_rate if funding is not None and not funding.empty else pd.Series(dtype=float)
    fidx=fs.index; fvals=fs.to_numpy(float)
    def close_trade(px, ts, reason):
        nonlocal side,entry,stop,target,risk,entry_i,entry_ts
        exitpx=px*(1-side*SLIP/10000)
        gross=side*(exitpx-entry)/entry
        a=fidx.searchsorted(entry_ts,"left"); b=fidx.searchsorted(ts,"right")
        fcost=float(np.sum(fvals[a:b])*side) if b>a else 0.
        net=gross-2*FEE/10000-fcost
        trades.append((symbol,tf,model,side,str(entry_ts),str(ts),entry,exitpx,stop,target,reason,
                       gross*10000,2*FEE,fcost*10000,net*10000,net/(risk/entry) if risk else 0.))
        side=0; entry=stop=target=risk=0.; entry_i=-1; entry_ts=None
    last_sample_i=-1
    for i in range(1,len(d)):
        ts=ix[i]
        if ts >= sample_end: break
        last_sample_i=i
        # Manage existing position
        if side:
            held=i-entry_i
            sh=(l[i]<=stop) if side==1 else (h[i]>=stop)
            th=(h[i]>=target) if side==1 else (l[i]<=target)
            if sh or th or held>=hold:
                if sh:
                    px=o[i] if ((side==1 and o[i]<stop) or (side==-1 and o[i]>stop)) else stop; why="STOP"
                elif th: px=target; why="TARGET"
                else: px=c[i]; why="TIME"
                close_trade(float(px),ts,why)
        # Arm signal from immediately preceding completed candle
        if not side and pending is None and sig[i-1] and ix[i-1] >= sample_start:
            s=int(sig[i-1]); hi=float(h[i-1]); lo=float(l[i-1]); span=hi-lo
            level=hi-retrace*span if s==1 else lo+retrace*span
            pending=(s,level,lo if s==1 else hi,i-1)
        if not side and pending is not None:
            s,level,st,si=pending
            if i-si>pending_bars: pending=None
            else:
                touched=(l[i]<=level) if s==1 else (h[i]>=level)
                invalid=(o[i]<=st) if s==1 else (o[i]>=st)
                if touched and not invalid:
                    raw=min(o[i],level) if s==1 else max(o[i],level)
                    fill=raw*(1+s*SLIP/10000); r=abs(fill-st)
                    if r>0 and ((s==1 and st<fill) or (s==-1 and st>fill)):
                        side=s; entry=fill; stop=st; risk=r; target=fill+s*rr*r; entry_i=i; entry_ts=ts
                        # If stop/target both touch on fill bar, assume stop.
                        sh=(l[i]<=stop) if s==1 else (h[i]>=stop)
                        th=(h[i]>=target) if s==1 else (l[i]<=target)
                        if sh or th:
                            if sh:
                                px=o[i] if ((s==1 and o[i]<stop) or (s==-1 and o[i]>stop)) else stop; why="STOP"
                            else: px=target; why="TARGET"
                            close_trade(float(px),ts,why)
                    pending=None
    if side and last_sample_i>=0:
        close_trade(float(c[last_sample_i]),ix[last_sample_i],"END_OF_SAMPLE")
    cols=["symbol","timeframe","model","side","entry_time","exit_time","entry","exit","stop","target","exit_reason","gross_bps","fees_bps","funding_cost_bps","net_bps","net_R"]
    return pd.DataFrame(trades,columns=cols)

def summarize(t):
    if t.empty: return dict(trades=0,win_rate_pct=0.,profit_factor=0.,net_bps=0.,avg_net_bps=0.,max_drawdown_pct=0.)
    w=t.net_bps>0; gp=t.loc[w,"net_bps"].sum(); gl=-t.loc[~w,"net_bps"].sum()
    eq=(1+t.net_bps/10000).cumprod(); dd=(eq/eq.cummax()-1).min()*100
    return dict(trades=len(t),win_rate_pct=round(100*w.mean(),2),profit_factor=round(gp/gl,4) if gl>0 else None,
                net_bps=round(t.net_bps.sum(),2),avg_net_bps=round(t.net_bps.mean(),3),max_drawdown_pct=round(dd,3),
                avg_net_R=round(t.net_R.mean(),3))

def main():
    out=Path("artifacts/model3_full_retracement"); out.mkdir(parents=True,exist_ok=True)
    raw={}; funds={}; quality=[]
    for s in SYMBOLS:
        raw[s]=acquire_klines(s,START,END)
        funds[s]=acquire_funding(s,START,END)
        quality.append(quality_report(raw[s],s))
    (out/"data_quality.json").write_text(json.dumps(quality,indent=2))
    # Prepared indicators cached once per symbol/timeframe/ATR period.
    prep={}
    for s in SYMBOLS:
      for tf in TIMEFRAMES:
       for ap in ATR_PERIODS:
        prep[s,tf,ap]=prepare(raw[s],tf,ap)
    # Core parameter grid without RR: 3*6*3*3*12*4*2 = 15,552 setups.
    # Each setup is evaluated at all five RR values and all chronological samples.
    summaries=[]; trade_parts=[]
    configs=list(product(TIMEFRAMES,ATR_PERIODS,IMPULSE_ATR,ADX_BANDS,RSI_MODES,RETRACEMENTS))
    total=len(configs)*len(RRS)*len(PENDING_BARS)*len(SYMBOLS)
    print(f"CONFIGS={len(configs)} ASSET_TESTS={total} PERIODS={len(CUTS)}")
    n=0
    for tf,ap,mult,(alo,ahi),mode,ret in configs:
      model=f"M3_{tf}_A{ap}_I{mult}_ADX{alo}-{ahi}_RSI{mode}_RET{ret:.3f}"
      for s in SYMBOLS:
        d=prep[s,tf,ap]; sig=make_impulses(d,mult,alo,ahi,mode)
        for rr in RRS:
          for pending_bars in PENDING_BARS:
           for sample,ss,se in CUTS:
            t=run_limit_bt(d,funds[s],sig,ret,rr,s,tf,model,pd.Timestamp(ss,tz="UTC"),pd.Timestamp(se,tz="UTC"),pending_bars)
            if len(t):
                t["sample"]=sample; t["atr_period"]=ap; t["impulse_atr"]=mult
                t["adx_min"]=alo; t["adx_max"]=ahi; t["rsi_mode"]=mode; t["retracement"]=ret; t["rr"]=rr; t["pending_bars"]=pending_bars
                trade_parts.append(t)
                met=summarize(t)
            else: met=summarize(t)
            summaries.append(dict(model=model,symbol=s,timeframe=tf,atr_period=ap,impulse_atr=mult,
              adx_min=alo,adx_max=ahi,rsi_mode=mode,retracement=ret,rr=rr,pending_bars=pending_bars,sample=sample,**met))
        n+=1
        if n%100==0: print(f"PROGRESS {n}/{len(configs)*len(SYMBOLS)}")
    sm=pd.DataFrame(summaries); sm.to_csv(out/"all_configuration_metrics.csv",index=False)
    if trade_parts: pd.concat(trade_parts,ignore_index=True).to_csv(out/"all_trades.csv",index=False)
    # Rank by combined forward validation only after requiring adequate trade count.
    fwd=sm[sm.sample.isin(["OOS_2024","OOS_2025","OOS_2026_YTD"])].copy()
    agg=fwd.groupby(["model","timeframe","atr_period","impulse_atr","adx_min","adx_max","rsi_mode","retracement","rr"],dropna=False).agg(
      trades=("trades","sum"),net_bps=("net_bps","sum"),win_rate_pct=("win_rate_pct","mean"),
      pf=("profit_factor","mean"),years_positive=("net_bps",lambda x:int((x>0).sum()))).reset_index()
    agg=agg[agg.trades>=60].sort_values(["years_positive","net_bps"],ascending=False)
    agg.to_csv(out/"forward_robustness_ranking.csv",index=False)
    # Best levels by forward net results, grouped across the full parameter grid.
    level=fwd.groupby(["retracement","rr"]).agg(trades=("trades","sum"),net_bps=("net_bps","sum"),
      mean_win_rate=("win_rate_pct","mean"),mean_pf=("profit_factor","mean")).reset_index()
    level.to_csv(out/"retracement_rr_level_comparison.csv",index=False)
    manifest={"status":"completed","symbols":list(SYMBOLS),"timeframes":list(TIMEFRAMES),
      "atr_periods":list(ATR_PERIODS),"impulse_atr":list(IMPULSE_ATR),"adx_bands":[list(x) for x in ADX_BANDS],
      "rsi_modes":list(RSI_MODES),"retracements":list(RETRACEMENTS),"rr":list(RRS),"pending_bars":list(PENDING_BARS),
      "configurations":len(configs),"asset_configurations":len(configs)*len(SYMBOLS),
      "periods":[x[0] for x in CUTS],"fee_bps_per_side":FEE,"slippage_bps_per_side":SLIP,
      "pending_valid_bars":list(PENDING_BARS),"entry_rule":"exact retracement price; touch on any subsequent bar; no confirmation candle",
      "funding":"official Binance Vision fundingRate archives"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("TOP FORWARD CANDIDATES"); print(agg.head(30).to_string(index=False))
    print("RETRACEMENT LEVEL SUMMARY"); print(level.to_string(index=False))

if __name__=="__main__": main()
