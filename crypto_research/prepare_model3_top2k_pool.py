from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

PARAMS=['timeframe','atr_period','impulse_atr','adx_min','adx_max','rsi_mode','retracement','rr','pending_bars']
BASE_FIELDS=['model_3x','pf_rank','group_id','symbol','timeframe','atr_period','impulse_atr','adx_min','adx_max','rsi_mode','retracement','rr','pending_bars','robust_score']

def broad_family(r):
    if r.adx_min==20 and r.adx_max==35 and r.retracement>=.80: return '3A'
    if r.adx_min==20 and r.adx_max==35 and r.retracement<=.786 and r.rsi_mode=='none': return '3B'
    if r.adx_min==18 and r.adx_max==30 and r.impulse_atr>=2.2 and r.retracement<=.82: return '3C'
    return '3X'

def baseline_family(r):
    if (r.adx_min==20 and r.adx_max==35 and r.impulse_atr in [2.0,2.2,2.4,2.5,2.7] and r.retracement in [.80,.82,.85] and r.rsi_mode in ['exact','none'] and r.rr in [2.,3.,4.,5.] and r.pending_bars==3): return '3A'
    if (r.adx_min==20 and r.adx_max==35 and r.impulse_atr in [1.8,2.0,2.2,2.4,2.5,2.7] and r.retracement in [.75,.786] and r.rsi_mode=='none' and r.rr==2. and r.pending_bars==3): return '3B'
    if (r.adx_min==18 and r.adx_max==30 and r.impulse_atr in [2.2,2.4,2.5,2.7] and r.retracement in [.75,.786,.82] and r.rsi_mode in ['exact','none'] and r.rr in [2.,3.] and r.pending_bars==3): return '3C'
    return None

def build_pool(source_csv,outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(source_csv,nrows=2000)
    if len(d)!=2000: raise RuntimeError(f'Expected 2,000 ranked rows, found {len(d)}')
    if not d.pf_rank.is_monotonic_increasing: d=d.sort_values('pf_rank')
    if (d.oos_profit_factor<=1).any(): raise RuntimeError('Top-2,000 source pool unexpectedly includes PF<=1 rows')
    rows=[]
    for _,g in d.groupby(PARAMS,sort=False,dropna=False):
        r=g.sort_values('pf_rank').iloc[0].copy(); r['candidate_family']=broad_family(r)
        r['source_symbols']=';'.join(sorted(g.symbol.unique())); r['source_rows']=len(g)
        r['best_source_rank']=int(g.pf_rank.min()); rows.append(r)
    pool=pd.DataFrame(rows).sort_values('best_source_rank').reset_index(drop=True)
    pool.insert(0,'candidate_id',[f'T2K_{i:04d}' for i in range(1,len(pool)+1)])
    if len(pool)!=1853: raise RuntimeError(f'Expected 1,853 unique parameter sets, found {len(pool)}')
    pool.to_csv(out/'model3_top2000_unique_candidate_pool.csv',index=False)
    btc=d[(d.symbol=='BTCUSDT')&(d.timeframe=='1h')].copy(); btc['family']=btc.apply(baseline_family,axis=1)
    btc=btc[btc.family.notna()]
    net=btc[['oos_2024_net_profit_bps','oos_2025_net_profit_bps','oos_2026_ytd_net_profit_bps']]
    pf=btc[['oos_2024_profit_factor','oos_2025_profit_factor','oos_2026_ytd_profit_factor']]
    keep=(btc.oos_trades>=60)&(net>0).all(axis=1)&(pf>=1.1).all(axis=1)
    btc=btc[keep]
    baseline={fam:[] for fam in ['3A','3B','3C']}
    for _,r in btc.iterrows():
        annual_pf=r[['oos_2024_profit_factor','oos_2025_profit_factor','oos_2026_ytd_profit_factor']].astype(float)
        annual_net=r[['oos_2024_net_profit_bps','oos_2025_net_profit_bps','oos_2026_ytd_net_profit_bps']].astype(float)
        robust=.25*float(r.oos_profit_factor)+.50*float(annual_pf.median())+.25*float(annual_pf.min())+.10*float(__import__('math').log1p(r.oos_trades))+.10*int((annual_net>0).sum())-.025*abs(float(r.oos_max_drawdown_pct))
        item={k:(float(r[k]) if k in ['impulse_atr','retracement','rr'] else int(r[k]) if k in ['pf_rank','group_id','atr_period','adx_min','adx_max','pending_bars'] else str(r[k])) for k in ['pf_rank','group_id','symbol','timeframe','atr_period','impulse_atr','adx_min','adx_max','rsi_mode','retracement','rr','pending_bars']}
        item['model_3x']=str(r.family); item['robust_score']=float(robust)
        baseline[r.family].append(item)
    counts={k:len(v) for k,v in baseline.items()}
    if counts!={'3A':21,'3B':8,'3C':17}: raise RuntimeError(f'Frozen BTC baseline mismatch: {counts}')
    def param_key(r):
        return (str(r['timeframe']),int(r['atr_period']),round(float(r['impulse_atr']),8),
                int(r['adx_min']),int(r['adx_max']),str(r['rsi_mode']),
                round(float(r['retracement']),8),round(float(r['rr']),8),int(r['pending_bars']))
    pool_ids={param_key(r):str(r['candidate_id']) for _,r in pool.iterrows()}
    baseline_ids=[]
    for fam,arr in baseline.items():
        for item in arr:
            key=param_key(item)
            if key not in pool_ids:
                raise RuntimeError(f'Frozen BTC baseline config missing from top-2,000 pool: {key}')
            item['candidate_id']=pool_ids[key]
            baseline_ids.append(pool_ids[key])
    if len(set(baseline_ids))!=46:
        raise RuntimeError(f'Frozen BTC baseline candidate IDs are not unique: {len(set(baseline_ids))}')
    (out/'model3_btc_frozen_baseline_46.json').write_text(json.dumps(baseline,indent=2))
    manifest={'source_rows':2000,'unique_parameter_sets':len(pool),'source_symbols':sorted(d.symbol.unique().tolist()),'candidate_family_counts':pool.candidate_family.value_counts().to_dict(),'frozen_btc_baseline_counts':counts,'frozen_btc_baseline_candidate_ids':baseline_ids,'rank_source':'all_eligible_ranked_by_pf.csv','rank_window':'2024-2026 YTD OOS on BTC/ETH'}
    (out/'top2k_pool_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); build_pool(a.source,a.out)