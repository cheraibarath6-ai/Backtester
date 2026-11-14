from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from .data import acquire_klines, acquire_funding
from .sweep_m3_full_retracement import prepare, make_impulses

HERE=Path(__file__).resolve().parent
POOL=HERE/'configs'/'model3_top2000_unique_candidate_pool.csv'
BASELINE=HERE/'configs'/'model3_btc_frozen_baseline_46.json'
SYMBOLS=('BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','LINKUSDT','LTCUSDT','AVAXUSDT')
PERIODS=(('TRAIN_2021_2023','2021-01-01','2024-01-01'),('OOS_2024','2024-01-01','2025-01-01'),('OOS_2025','2025-01-01','2026-01-01'),('OOS_2026_YTD','2026-01-01','2026-11-01'))
HOLD={'15m':96,'30m':48,'1h':24}; FEE=5.; SLIP=2.


def funding_quality(funding,symbol):
    if funding.empty:
        raise RuntimeError(f'Official Binance funding data is empty for {symbol}')
    gaps=funding.index.to_series().diff().dropna()
    max_gap=float(gaps.max().total_seconds()/3600) if len(gaps) else 0.
    if 'funding_interval_hours' in funding:
        intervals=pd.to_numeric(funding.funding_interval_hours,errors='coerce').fillna(8.).to_numpy(float)
    else:
        intervals=np.full(len(funding),8.,dtype=float)
    suspicious=[]
    missing_estimate=0
    for i,delta in enumerate(gaps.to_numpy()):
        hours=float(delta/np.timedelta64(1,'h'))
        interval=max(float(intervals[i]),float(intervals[i+1]))
        if hours>1.75*interval:
            missing=max(1,int(round(hours/interval))-1)
            missing_estimate+=missing
            suspicious.append({'from':str(funding.index[i]),'to':str(funding.index[i+1]),
                               'gap_hours':hours,'expected_interval_hours':interval,'estimated_missing_events':missing})
    expected=len(funding)+missing_estimate
    return {'symbol':symbol,'source':funding.attrs.get('source','unknown'),'rows':len(funding),
            'start':str(funding.index.min()),'end':str(funding.index.max()),
            'median_interval_hours':float(np.median(intervals)),
            'max_gap_hours':max_gap,'estimated_missing_events':int(missing_estimate),
            'coverage_pct':100.*len(funding)/expected if expected else 0.,
            'suspicious_gap_count':len(suspicious),'suspicious_gaps':suspicious[:25]}


def candle_quality(raw,symbol):
    idx=raw.index.sort_values(); gaps=idx.to_series().diff().dropna(); expected=int((idx[-1]-idx[0]).total_seconds()/60)+1
    missing=expected-len(idx); max_gap=float(gaps.max().total_seconds()/60) if len(gaps) else 0.
    intervals=[]
    for gap_end,delta in gaps[gaps>pd.Timedelta(minutes=1)].items():
        intervals.append({'first_missing':str(gap_end-delta+pd.Timedelta(minutes=1)),
                          'last_missing':str(gap_end-pd.Timedelta(minutes=1)),
                          'missing_minutes':int(delta.total_seconds()/60)-1})
    return {'symbol':symbol,'rows':len(raw),'actual_start':str(idx[0]),'actual_end':str(idx[-1]),
            'missing_minutes':int(missing),'coverage_pct':100*len(raw)/expected,'max_gap_minutes':max_gap,
            'gap_count':len(intervals),'missing_intervals':intervals[:25],
            'daily_backfill_days_attempted':int(raw.attrs.get('daily_backfill_days_attempted',0)),
            'daily_backfill_days_loaded':int(raw.attrs.get('daily_backfill_days_loaded',0))}


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
    if len(net)==0:
        return {'trades':0,'wins':0,'losses':0,'win_rate_pct':0.,'profit_factor':None,'net_bps':0.,
                'net_winning_bps':0.,'net_losing_bps':0.,'max_drawdown_pct':0.,
                'long_trades':0,'short_trades':0,'stop_exits':0,'target_exits':0}
    w=net>0; gp=float(net[w].sum()); gl=float(-net[~w].sum())
    eq=np.cumprod(1+net/10000); dd=float(np.min(eq/np.maximum.accumulate(eq)-1)*100)
    return {'trades':int(len(net)),'wins':int(w.sum()),'losses':int((~w).sum()),
            'win_rate_pct':float(w.mean()*100),'profit_factor':gp/gl if gl else None,
            'net_bps':float(net.sum()),'net_winning_bps':gp,'net_losing_bps':gl,
            'max_drawdown_pct':dd,'long_trades':int((sides==1).sum()),
            'short_trades':int((sides==-1).sum()),'stop_exits':int((reasons==1).sum()),
            'target_exits':int((reasons==2).sum())}


def run_asset(symbol,out):
    if symbol not in SYMBOLS: raise ValueError(f'Unsupported symbol: {symbol}')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    print(f'{symbol}: downloading Binance USD-M 1m candles',flush=True)
    raw=acquire_klines(symbol,'2021-01-01','2026-11-01',root=out/'data')
    q=candle_quality(raw,symbol)
    (out/'data_quality.json').write_text(json.dumps(q,indent=2))
    if q['missing_minutes']!=0 or q['coverage_pct']<100:
        raise RuntimeError(f'Candle gaps remain after official daily backfill; refusing incomplete replay: {q}')

    funding=acquire_funding(symbol,'2021-01-01','2026-11-01',root=out/'data')
    fq=funding_quality(funding,symbol)
    (out/'funding_quality.json').write_text(json.dumps(fq,indent=2))
    if fq['suspicious_gap_count']>0 or fq['coverage_pct']<99.9:
        raise RuntimeError(f'Official Binance funding coverage is incomplete; refusing assumed-zero funding: {fq}')

    pool=pd.read_csv(POOL)
    if len(pool)!=1853 or pool.candidate_id.nunique()!=1853:
        raise RuntimeError(f'Expected exactly 1,853 unique candidates; got rows={len(pool)} unique={pool.candidate_id.nunique()}')
    baseline=json.loads(BASELINE.read_text())
    baseline_map={str(x['candidate_id']):fam for fam,arr in baseline.items() for x in arr}
    if len(baseline_map)!=46:
        raise RuntimeError(f'Frozen BTC baseline must contain exactly 46 unique candidate IDs; got {len(baseline_map)}')

    prep={}; sigcache={}; contexts={}; rows=[]
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
            rows.append({'candidate_id':r.candidate_id,'source_rank':int(r.best_source_rank),'symbol':symbol,
                         'family':r.candidate_family,'baseline_member':str(r.candidate_id) in baseline_map,
                         'baseline_family':baseline_map.get(str(r.candidate_id),''),'timeframe':tf,'atr_period':ap,
                         'impulse_atr':float(r.impulse_atr),'adx_min':int(r.adx_min),'adx_max':int(r.adx_max),
                         'rsi_mode':r.rsi_mode,'retracement':float(r.retracement),'rr':float(r.rr),
                         'pending_bars':int(r.pending_bars),'period':period,**stats(net,sd,rs)})
        if (j+1)%100==0: print(f'{symbol}: {j+1}/{len(pool)} configs',flush=True)

    m=pd.DataFrame(rows)
    expected_rows=1853*len(PERIODS)
    if len(m)!=expected_rows or m.candidate_id.nunique()!=1853:
        raise RuntimeError(f'{symbol}: incomplete candidate replay rows={len(m)} unique={m.candidate_id.nunique()}')
    per_period=m.groupby('period').candidate_id.nunique().to_dict()
    if any(per_period.get(p,0)!=1853 for p,_,_ in PERIODS):
        raise RuntimeError(f'{symbol}: each of four periods must contain all 1,853 candidates: {per_period}')
    m.to_csv(out/f'{symbol}_all_metrics.csv',index=False)

    tr=m[m.period=='TRAIN_2021_2023'].copy()
    if symbol=='BTCUSDT':
        selected=tr[tr.candidate_id.astype(str).isin(baseline_map)].copy()
        selected['family']=selected.candidate_id.astype(str).map(baseline_map)
        selected['selection_basis']='FROZEN_BTC_BASELINE_46'
        if len(selected)!=46: raise RuntimeError(f'BTC frozen baseline replay mismatch: expected 46, got {len(selected)}')
    else:
        tr=tr[(tr.trades>=60)&(tr.profit_factor>=1)&(tr.net_bps>0)].copy()
        tr['dev_score']=.6*tr.profit_factor+.1*np.log1p(tr.trades)+.2*np.clip(tr.net_bps/10000,-2,2)-.02*tr.max_drawdown_pct.abs()
        selected=pd.concat([tr[tr.family==fam].sort_values(['dev_score','profit_factor','net_bps'],ascending=False).head(15)
                             for fam in ('3A','3B','3C','3X')],ignore_index=True)
        selected['selection_basis']='ASSET_TRAIN_2021_2023'
    selected.to_csv(out/f'{symbol}_selected_configs.csv',index=False)
    summary={'symbol':symbol,'candidate_configs':len(pool),'candidate_ids':pool.candidate_id.nunique(),
             'rows_per_period':per_period,'selected_development_configs':len(selected),
             'selected_by_family':selected.family.value_counts().to_dict() if len(selected) else {},
             'selection_basis':'frozen original 46 BTC configurations' if symbol=='BTCUSDT' else '2021-2023 development only; up to 15 per family',
             'candle_quality':q,'funding_quality':fq}
    (out/f'{symbol}_summary.json').write_text(json.dumps(summary,indent=2))
    print(f'{symbol}: completed {len(m)} candidate-period rows; selected {len(selected)} configs',flush=True)


def grouped_performance(frame,group_cols):
    g=frame.groupby(group_cols,dropna=False).agg(
        configurations=('candidate_id','nunique'),trades=('trades','sum'),wins=('wins','sum'),losses=('losses','sum'),
        net_winning_bps=('net_winning_bps','sum'),net_losing_bps=('net_losing_bps','sum'),
        net_bps=('net_bps','sum'),worst_config_drawdown_pct=('max_drawdown_pct','min'),
        median_config_drawdown_pct=('max_drawdown_pct','median'),positive_configs=('net_bps',lambda x:int((x>0).sum())),
        long_trades=('long_trades','sum'),short_trades=('short_trades','sum')).reset_index()
    g['win_rate_pct']=np.where(g.trades>0,100.*g.wins/g.trades,0.)
    g['profit_factor']=g.net_winning_bps/g.net_losing_bps.replace({0:np.nan})
    return g


def aggregate(inp,out):
    inp=Path(inp); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    pool=pd.read_csv(POOL)
    if len(pool)!=1853 or pool.candidate_id.nunique()!=1853:
        raise RuntimeError('Frozen candidate pool is not exactly 1,853 unique configurations')
    expected_ids=set(pool.candidate_id.astype(str))
    files=sorted(inp.glob('**/*_all_metrics.csv'))
    found={p.name.replace('_all_metrics.csv','') for p in files}
    missing=set(SYMBOLS)-found
    if missing: raise RuntimeError(f'Missing asset metric files: {sorted(missing)}')

    metric_parts=[]; summaries=[]; completeness={}
    for symbol in SYMBOLS:
        fp=next(p for p in files if p.name==f'{symbol}_all_metrics.csv')
        part=pd.read_csv(fp)
        if part.candidate_id.nunique()!=1853 or set(part.candidate_id.astype(str))!=expected_ids:
            raise RuntimeError(f'{symbol}: candidate ID coverage mismatch')
        counts=part.groupby('period').candidate_id.nunique().to_dict()
        if any(counts.get(period,0)!=1853 for period,_,_ in PERIODS):
            raise RuntimeError(f'{symbol}: expected 1,853 unique candidates in every period, got {counts}')
        if part.duplicated(['candidate_id','period']).any():
            raise RuntimeError(f'{symbol}: duplicate candidate-period rows detected')
        sp=next(iter(inp.glob(f'**/{symbol}_summary.json')),None)
        if sp is None: raise RuntimeError(f'{symbol}: missing summary/data-quality manifest')
        sm=json.loads(sp.read_text())
        q=sm['candle_quality']; fq=sm['funding_quality']
        if q['missing_minutes']!=0 or q['coverage_pct']<100:
            raise RuntimeError(f'{symbol}: candle coverage failed: {q}')
        if not str(fq['source']).startswith('binance_vision_official_monthly_fundingRate'):
            raise RuntimeError(f'{symbol}: funding provenance is not official Binance Vision: {fq}')
        if fq['suspicious_gap_count']!=0 or fq['coverage_pct']<99.9:
            raise RuntimeError(f'{symbol}: official funding coverage failed: {fq}')
        if sm.get('candidate_configs')!=1853 or sm.get('candidate_ids')!=1853:
            raise RuntimeError(f'{symbol}: summary does not confirm 1,853 candidates')
        completeness[symbol]={'candidate_ids':part.candidate_id.nunique(),'rows':len(part),
                              'rows_per_period':counts,'candle_quality':q,'funding_quality':fq,
                              'selected_configs':sm.get('selected_development_configs',0)}
        summaries.append({'symbol':symbol,**q,**{f'funding_{k}':v for k,v in fq.items() if k not in ('suspicious_gaps','symbol')}})
        metric_parts.append(part)

    m=pd.concat(metric_parts,ignore_index=True)
    expected_rows=len(SYMBOLS)*1853*len(PERIODS)
    if len(m)!=expected_rows: raise RuntimeError(f'Expected {expected_rows} metric rows, got {len(m)}')
    m.to_csv(out/'TOP2K_ALL_ASSET_ALL_CONFIG_METRICS.csv',index=False)

    all_family=grouped_performance(m,['symbol','family','period'])
    all_family.to_csv(out/'TOP2K_MODEL_FAMILY_ALL_CANDIDATE_PERFORMANCE.csv',index=False)
    all_asset=grouped_performance(m,['symbol','period'])
    all_asset.to_csv(out/'TOP2K_ALL_CANDIDATE_ASSET_PERFORMANCE.csv',index=False)

    sel_files=sorted(inp.glob('**/*_selected_configs.csv'))
    if len(sel_files)!=len(SYMBOLS): raise RuntimeError(f'Expected 10 selected-config files, found {len(sel_files)}')
    selected=pd.concat([pd.read_csv(p) for p in sel_files],ignore_index=True)
    selected.to_csv(out/'ASSET_ORIENTED_SELECTED_CONFIGS.csv',index=False)
    keys=selected[['symbol','candidate_id','family','selection_basis']].drop_duplicates()
    so=m.drop(columns=['family']).merge(keys,on=['symbol','candidate_id'],how='inner',validate='many_to_one')
    so=so[so.period.str.startswith('OOS_')].copy()
    so.to_csv(out/'ASSET_ORIENTED_SELECTED_CONFIGS_WITH_OOS.csv',index=False)

    model_perf=grouped_performance(so,['symbol','family','period'])
    model_perf.to_csv(out/'ASSET_ORIENTED_MODEL_PERFORMANCE.csv',index=False)
    asset_perf=grouped_performance(so,['symbol','period'])
    asset_perf.to_csv(out/'ASSET_ORIENTED_ASSET_PERFORMANCE.csv',index=False)
    consistency=model_perf.pivot_table(index=['symbol','family'],columns='period',
        values=['trades','win_rate_pct','profit_factor','net_bps','worst_config_drawdown_pct'],aggfunc='first')
    consistency.columns=[f'{metric}_{period}' for metric,period in consistency.columns]
    consistency=consistency.reset_index()
    net_cols=[c for c in consistency.columns if c.startswith('net_bps_OOS_')]
    consistency['positive_oos_periods']=sum((consistency[c]>0).astype(int) for c in net_cols)
    consistency.to_csv(out/'ASSET_ORIENTED_YEARLY_CONSISTENCY.csv',index=False)

    baseline=m[(m.symbol=='BTCUSDT')&(m.baseline_member==True)].copy()
    if baseline.candidate_id.nunique()!=46: raise RuntimeError(f'BTC frozen baseline should have 46 configs, got {baseline.candidate_id.nunique()}')
    baseline_perf=grouped_performance(baseline,['baseline_family','period'])
    baseline_perf.to_csv(out/'FROZEN_BTC_BASELINE_PERFORMANCE.csv',index=False)

    oos=m[m.period.str.startswith('OOS_')].groupby(['symbol','period']).agg(
        configs=('candidate_id','nunique'),positive_net_configs=('net_bps',lambda x:int((x>0).sum())),
        median_pf=('profit_factor','median'),median_trades=('trades','median'),
        total_net_bps=('net_bps','sum')).reset_index()
    oos.to_csv(out/'TOP2K_ASSET_OOS_SUMMARY.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'DATA_AND_FUNDING_COVERAGE.csv',index=False)

    baseline_json=json.loads(BASELINE.read_text())
    by={sym:[] for sym in SYMBOLS}
    for _,r in selected.iterrows():
        by[r.symbol].append({k:(int(r[k]) if k in ('source_rank','atr_period','adx_min','adx_max','pending_bars')
            else float(r[k]) if k in ('impulse_atr','retracement','rr')
            else str(r[k])) for k in ('candidate_id','source_rank','family','timeframe','atr_period','impulse_atr',
                                      'adx_min','adx_max','rsi_mode','retracement','rr','pending_bars','selection_basis')})
    config_doc={'btc_frozen_baseline_46':baseline_json,'asset_oriented_candidates':by,
                'btc_candidates_are_frozen_baseline_only':True,
                'other_asset_selection':'2021-2023 development only; minimum 60 trades, PF >= 1, positive net bps; up to 15 per family',
                'candidate_pool':'exact top 2,000 source rows deduplicated to 1,853 unique parameter configurations',
                'oos_caveat':'The source top-2,000 pool was ranked using 2024-2026 BTC/ETH results; cross-asset OOS is exploratory and not an untouched blind test.'}
    (out/'ASSET_ORIENTED_MODEL3_CONFIGS.json').write_text(json.dumps(config_doc,indent=2))
    report={'status':'completed','assets':list(SYMBOLS),'candidate_configs_per_asset':1853,
            'periods':[p[0] for p in PERIODS],'expected_metric_rows':expected_rows,'actual_metric_rows':len(m),
            'asset_completeness':completeness,'btc_frozen_baseline_configs':46,
            'selected_configs_total':len(selected),'selection_basis':'BTC frozen baseline; other assets selected on 2021-2023 development',
            'oos_caveat':config_doc['oos_caveat']}
    (out/'FINAL_COMPLETENESS.json').write_text(json.dumps(report,indent=2))
    print('TEN-ASSET COMPLETENESS VERIFIED')
    print(json.dumps({k:{'candidate_ids':v['candidate_ids'],'rows':v['rows'],'selected_configs':v['selected_configs']} for k,v in completeness.items()},indent=2))
    print('ASSET-ORIENTED MODEL PERFORMANCE')
    print(model_perf.to_string(index=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--symbol'); ap.add_argument('--out',default='asset_results'); ap.add_argument('--aggregate',action='store_true'); ap.add_argument('--input',default='input-artifacts'); a=ap.parse_args()
    aggregate(a.input,a.out) if a.aggregate else run_asset(a.symbol,a.out)