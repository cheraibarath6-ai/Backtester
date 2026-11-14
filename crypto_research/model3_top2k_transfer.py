from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from numba import njit
from .data import acquire_klines
from .sweep_m3_full_retracement import prepare, make_impulses

HERE=Path(__file__).resolve().parent
POOL=HERE/'configs'/'model3_top2000_unique_candidate_pool.csv'
BASELINE=HERE/'configs'/'model3_btc_frozen_baseline_46.json'
SYMBOLS=('BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','LTCUSDT','AVAXUSDT')
PERIODS=(('TRAIN_2021_2023','2021-01-01','2024-01-01'),('OOS_2024','2024-01-01','2025-01-01'),('OOS_2025','2025-01-01','2026-01-01'),('OOS_2026_YTD','2026-01-01','2026-11-01'))
HOLD={'15m':96,'30m':48,'1h':24}; FEE=5.; SLIP=2.


def official_funding(symbol,start='2021-01-01',end='2026-11-01'):
    # Official Binance USD-M endpoint only. No third-party fallback or assumed-zero funding.
    cursor=int(pd.Timestamp(start,tz='UTC').timestamp()*1000); end_ms=int(pd.Timestamp(end,tz='UTC').timestamp()*1000); rows=[]
    while cursor<end_ms:
        r=requests.get('https://fapi.binance.com/fapi/v1/fundingRate',params={'symbol':symbol,'startTime':cursor,'endTime':end_ms,'limit':1000},timeout=40)
        r.raise_for_status(); batch=r.json()
        if not batch: break
        rows.extend(batch); nxt=int(batch[-1]['fundingTime'])+1
        if nxt<=cursor: raise RuntimeError(f'Funding pagination stalled for {symbol}')
        cursor=nxt; time.sleep(.12)
    if not rows: raise RuntimeError(f'Official Binance funding unavailable for {symbol}')
    f=pd.DataFrame(rows); f['funding_time']=pd.to_datetime(f.fundingTime.astype('int64'),unit='ms',utc=True); f['funding_rate']=pd.to_numeric(f.fundingRate,errors='coerce')
    f=f.dropna(subset=['funding_rate']).drop_duplicates('funding_time').set_index('funding_time')[['funding_rate']].sort_index()
    gaps=f.index.to_series().diff().dropna(); max_gap=float(gaps.max().total_seconds()/3600) if len(gaps) else 0.
    if max_gap>48: raise RuntimeError(f'Official funding gap {max_gap:.1f}h for {symbol}; fail closed')
    f.attrs.update(source='binance_official_fapi_fundingRate',max_gap_hours=max_gap)
    return f


def candle_quality(raw,symbol):
    idx=raw.index.sort_values(); gaps=idx.to_series().diff().dropna(); expected=int((idx[-1]-idx[0]).total_seconds()/60)+1
    missing=expected-len(idx); max_gap=float(gaps.max().total_seconds()/60) if len(gaps) else 0.
    return {'symbol':symbol,'rows':len(raw),'actual_start':str(idx[0]),'actual_end':str(idx[-1]),'missing_minutes':int(missing),'coverage_pct':100*len(raw)/expected,'max_gap_minutes':max_gap}


@njit(cache=True)
def bt(ts,o,h,l,c,sig,fts,fprefix,start_ns,end_ns,retrace,rr,pending_bars,max_hold):
    a=np.searchsorted(ts,start_ns,'left'); b=np.searchsorted(ts,end_ns,'left'); a=max(1,a)
    cap=max(2,b-a+2); net=np.empty(cap); sideout=np.empty(cap,dtype=np.int8); entryout=np.empty(cap); exitout=np.empty(cap); reason=np.empty(cap,dtype=np.int8); ntr=0
    side=0; entry=stop=target=risk=0.; ei=-1; ps=0; plevel=pstop=0.; pi=-1
    for i in range(a,b):
        if side!=0:
            held=i-ei; sh=(l[i]<=stop) if side==1 else (h[i]>=stop); th=(h[i]>=target) if side==1 else (l[i]<=target)
            if sh or th or held>=max_hold:
                if sh: px=o[i] if ((side==1 and o[i]<stop) or (side==-1 and o[i]>stop)) else stop; why=1
                elif th: px=target; why=2
                else: px=c[i]; why=3
                xp=px*(1-side*SLIP/10000); gross=side*(xp-entry)/entry
                fa=np.searchsorted(fts,ts[ei],'left'); fb=np.searchsorted(fts,ts[i],'right'); fc=(fprefix[fb]-fprefix[fa])*side
                net[ntr]=(gross-2*FEE/10000-fc)*10000; sideout[ntr]=side; entryout[ntr]=entry; exitout[ntr]=xp; reason[ntr]=why; ntr+=1
                side=0; entry=stop=target=risk=0.; ei=-1
        if side==0 and ps==0 and sig[i-1]!=0 and ts[i-1]>=start_ns:
            s=int(sig[i-1]); span=h[i-1]-l[i-1]; plevel=h[i-1]-retrace*span if s==1 else l[i-1]+retrace*span; pstop=l[i-1] if s==1 else h[i-1]; ps=s; pi=i-1
        if side==0 and ps!=0:
            if i-pi>pending_bars: ps=0; pi=-1
            else:
                s=ps; touched=(l[i]<=plevel) if s==1 else (h[i]>=plevel); invalid=(o[i]<=pstop) if s==1 else (o[i]>=pstop)
                if invalid: ps=0; pi=-1
                elif touched:
                    raw=min(o[i],plevel) if s==1 else max(o[i],plevel); fill=raw*(1+s*SLIP/10000); r=abs(fill-pstop)
                    if r>0 and ((s==1 and pstop<fill) or (s==-1 and pstop>fill)):
                        side=s; entry=fill; stop=pstop; risk=r; target=fill+s*rr*r; ei=i
                        sh=(l[i]<=stop) if s==1 else (h[i]>=stop); th=(h[i]>=target) if s==1 else (l[i]<=target)
                        if sh or th:
                            if sh: px=o[i] if ((s==1 and o[i]<stop) or (s==-1 and o[i]>stop)) else stop; why=1
                            else: px=target; why=2
                            xp=px*(1-s*SLIP/10000); gross=s*(xp-entry)/entry; fa=np.searchsorted(fts,ts[ei],'left'); fb=np.searchsorted(fts,ts[i],'right'); fc=(fprefix[fb]-fprefix[fa])*s
                            net[ntr]=(gross-2*FEE/10000-fc)*10000; sideout[ntr]=s; entryout[ntr]=entry; exitout[ntr]=xp; reason[ntr]=why; ntr+=1
                            side=0; entry=stop=target=risk=0.; ei=-1
                    ps=0; pi=-1
    if side!=0 and b>a:
        i=b-1; xp=c[i]*(1-side*SLIP/10000); gross=side*(xp-entry)/entry; fa=np.searchsorted(fts,ts[ei],'left'); fb=np.searchsorted(fts,ts[i],'right'); fc=(fprefix[fb]-fprefix[fa])*side
        net[ntr]=(gross-2*FEE/10000-fc)*10000; sideout[ntr]=side; entryout[ntr]=entry; exitout[ntr]=xp; reason[ntr]=4; ntr+=1
    return net[:ntr],sideout[:ntr],reason[:ntr]


def stats(net,sides,reasons):
    if len(net)==0: return {'trades':0,'wins':0,'losses':0,'win_rate_pct':0.,'profit_factor':None,'net_bps':0.,'max_drawdown_pct':0.,'long_trades':0,'short_trades':0}
    w=net>0; gp=float(net[w].sum()); gl=float(-net[~w].sum()); eq=np.cumprod(1+net/10000); dd=float(np.min(eq/np.maximum.accumulate(eq)-1)*100)
    return {'trades':int(len(net)),'wins':int(w.sum()),'losses':int((~w).sum()),'win_rate_pct':float(w.mean()*100),'profit_factor':gp/gl if gl else None,'net_bps':float(net.sum()),'max_drawdown_pct':dd,'long_trades':int((sides==1).sum()),'short_trades':int((sides==-1).sum()),'stop_exits':int((reasons==1).sum()),'target_exits':int((reasons==2).sum())}


def run_asset(symbol,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); print(f'{symbol}: downloading Binance USD-M 1m candles',flush=True)
    raw=acquire_klines(symbol,'2021-01-01','2026-11-01',root=out/'data'); q=candle_quality(raw,symbol)
    if q['missing_minutes']!=0 or q['coverage_pct']<100: raise RuntimeError(f'Candle gaps; refusing incomplete replay: {q}')
    funding=official_funding(symbol); fq={'source':funding.attrs['source'],'rows':len(funding),'start':str(funding.index.min()),'end':str(funding.index.max()),'max_gap_hours':funding.attrs['max_gap_hours']}
    pool=pd.read_csv(POOL); prep={}; sigcache={}; contexts={}; rows=[]
    fts=funding.index.view('int64'); fr=funding.funding_rate.to_numpy(float); fp=np.r_[0.,np.cumsum(fr)]
    for j,r in pool.iterrows():
        tf=r.timeframe; ap=int(r.atr_period); key=(tf,ap)
        if key not in prep:
            d=prepare(raw,tf,ap); prep[key]=d
            ts=d.index.view('int64'); contexts[key]=(ts,*[d[x].to_numpy(float) for x in ('open','high','low','close')])
        d=prep[key]; sk=(tf,ap,float(r.impulse_atr),int(r.adx_min),int(r.adx_max),str(r.rsi_mode))
        if sk not in sigcache: sigcache[sk]=make_impulses(d,float(r.impulse_atr),int(r.adx_min),int(r.adx_max),str(r.rsi_mode))
        ts,o,h,l,c=contexts[key]; sig=sigcache[sk]
        for period,ps,pe in PERIODS:
            net,sd,rs=bt(ts,o,h,l,c,sig,fts,fp,pd.Timestamp(ps,tz='UTC').value,pd.Timestamp(pe,tz='UTC').value,float(r.retracement),float(r.rr),int(r.pending_bars),HOLD[tf])
            rows.append({'candidate_id':r.candidate_id,'source_rank':int(r.best_source_rank),'symbol':symbol,'family':r.candidate_family,'timeframe':tf,'atr_period':ap,'impulse_atr':float(r.impulse_atr),'adx_min':int(r.adx_min),'adx_max':int(r.adx_max),'rsi_mode':r.rsi_mode,'retracement':float(r.retracement),'rr':float(r.rr),'pending_bars':int(r.pending_bars),'period':period,**stats(net,sd,rs)})
        if (j+1)%100==0: print(f'{symbol}: {j+1}/{len(pool)} configs',flush=True)
    m=pd.DataFrame(rows); m.to_csv(out/f'{symbol}_all_metrics.csv',index=False)
    tr=m[m.period=='TRAIN_2021_2023'].copy(); tr=tr[(tr.trades>=60)&(tr.profit_factor>=1)&(tr.net_bps>0)]
    tr['dev_score']=.6*tr.profit_factor+.1*np.log1p(tr.trades)+.2*np.clip(tr.net_bps/10000,-2,2)-.02*tr.max_drawdown_pct.abs()
    selected=pd.concat([tr[tr.family==fam].sort_values(['dev_score','profit_factor','net_bps'],ascending=False).head(15) for fam in ('3A','3B','3C','3X')],ignore_index=True)
    selected.to_csv(out/f'{symbol}_selected_configs.csv',index=False)
    summary={'symbol':symbol,'candidate_configs':len(pool),'selected_development_configs':len(selected),'selected_by_family':selected.family.value_counts().to_dict() if len(selected) else {},'candle_quality':q,'funding_quality':fq}
    (out/f'{symbol}_summary.json').write_text(json.dumps(summary,indent=2)); print(f'{symbol}: selected {len(selected)} configs',flush=True)


def aggregate(inp,out):
    inp=Path(inp); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    files=sorted(inp.glob('**/*_all_metrics.csv')); found={p.name.replace('_all_metrics.csv','') for p in files}; missing=set(SYMBOLS)-found
    if missing: raise RuntimeError(f'Missing assets: {sorted(missing)}')
    m=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    if len(m)!=len(SYMBOLS)*1853*4: raise RuntimeError(f'Expected {len(SYMBOLS)*1853*4} metric rows, got {len(m)}')
    m.to_csv(out/'TOP2K_ALL_ASSET_ALL_CONFIG_METRICS.csv',index=False)
    sel_files=sorted(inp.glob('**/*_selected_configs.csv')); s=pd.concat([pd.read_csv(p) for p in sel_files],ignore_index=True)
    s.to_csv(out/'ASSET_ORIENTED_SELECTED_CONFIGS.csv',index=False)
    keys=s[['symbol','candidate_id']].drop_duplicates(); so=m[m.period.str.startswith('OOS_')].merge(keys,on=['symbol','candidate_id'])
    so.to_csv(out/'ASSET_ORIENTED_SELECTED_CONFIGS_WITH_OOS.csv',index=False)
    summary=m[m.period.str.startswith('OOS_')].groupby(['symbol','period']).agg(configs=('candidate_id','size'),positive_net_configs=('net_bps',lambda x:int((x>0).sum())),median_pf=('profit_factor','median'),median_trades=('trades','median')).reset_index()
    summary.to_csv(out/'TOP2K_ASSET_OOS_SUMMARY.csv',index=False)
    baseline=json.loads(BASELINE.read_text()); by={sym:[] for sym in SYMBOLS}
    for _,r in s.iterrows(): by[r.symbol].append({k:(int(r[k]) if k in ('source_rank','atr_period','adx_min','adx_max','pending_bars') else float(r[k]) if k in ('impulse_atr','retracement','rr') else str(r[k])) for k in ('candidate_id','source_rank','family','timeframe','atr_period','impulse_atr','adx_min','adx_max','rsi_mode','retracement','rr','pending_bars')})
    (out/'ASSET_ORIENTED_MODEL3_CONFIGS.json').write_text(json.dumps({'btc_frozen_baseline_46':baseline,'asset_oriented_candidates':by,'btc_candidates_are_research_only':True,'selection':'2021-2023 development only; 2024-2026 OOS are evaluation only'},indent=2))
    print(summary.to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--symbol'); ap.add_argument('--out',default='asset_results'); ap.add_argument('--aggregate',action='store_true'); ap.add_argument('--input',default='input-artifacts'); a=ap.parse_args()
    aggregate(a.input,a.out) if a.aggregate else run_asset(a.symbol,a.out)