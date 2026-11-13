from __future__ import annotations
import argparse, json
from itertools import product
from pathlib import Path
import pandas as pd
from .data import acquire_klines, acquire_funding
from .sweep_m3_full_retracement import (
    SYMBOLS,TIMEFRAMES,RETRACEMENTS,ATR_PERIODS,IMPULSE_ATR,ADX_BANDS,
    RSI_MODES,RRS,PENDING_BARS,START,END,CUTS,FEE,SLIP,
    make_impulses,run_limit_bt,summarize,prepare
)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard-index",type=int,required=True)
    p.add_argument("--shard-count",type=int,required=True)
    p.add_argument("--out",default="artifacts/shards")
    a=p.parse_args()
    if not (0 <= a.shard_index < a.shard_count):
        raise ValueError("invalid shard index/count")
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    raw={s:acquire_klines(s,START,END) for s in SYMBOLS}
    funds={s:acquire_funding(s,START,END) for s in SYMBOLS}
    prep={}
    for s in SYMBOLS:
      for tf in TIMEFRAMES:
       for ap in ATR_PERIODS:
        prep[s,tf,ap]=prepare(raw[s],tf,ap)
    configs=list(product(TIMEFRAMES,ATR_PERIODS,IMPULSE_ATR,ADX_BANDS,RSI_MODES,RETRACEMENTS))
    total_groups=len(configs)*len(SYMBOLS)
    selected=[gid for gid in range(total_groups) if gid % a.shard_count == a.shard_index]
    summaries=[]; trade_parts=[]
    print(f"SHARD {a.shard_index+1}/{a.shard_count} GROUPS={len(selected)} TOTAL_GROUPS={total_groups}",flush=True)
    for k,gid in enumerate(selected,1):
      ci,si=divmod(gid,len(SYMBOLS))
      tf,ap,mult,(alo,ahi),mode,ret=configs[ci]
      s=SYMBOLS[si]
      model=f"M3_{tf}_A{ap}_I{mult}_ADX{alo}-{ahi}_RSI{mode}_RET{ret:.3f}"
      d=prep[s,tf,ap]; sig=make_impulses(d,mult,alo,ahi,mode)
      for rr in RRS:
       for pending_bars in PENDING_BARS:
        for sample,ss,se in CUTS:
         t=run_limit_bt(d,funds[s],sig,ret,rr,s,tf,model,pd.Timestamp(ss,tz="UTC"),pd.Timestamp(se,tz="UTC"),pending_bars)
         if len(t):
          t["sample"]=sample; t["atr_period"]=ap; t["impulse_atr"]=mult
          t["adx_min"]=alo; t["adx_max"]=ahi; t["rsi_mode"]=mode
          t["retracement"]=ret; t["rr"]=rr; t["pending_bars"]=pending_bars
          t["group_id"]=gid; trade_parts.append(t)
         met=summarize(t)
         summaries.append(dict(group_id=gid,model=model,symbol=s,timeframe=tf,atr_period=ap,
           impulse_atr=mult,adx_min=alo,adx_max=ahi,rsi_mode=mode,retracement=ret,
           rr=rr,pending_bars=pending_bars,sample=sample,**met))
      if k%8==0 or k==len(selected):
       print(f"SHARD_PROGRESS {a.shard_index} {k}/{len(selected)}",flush=True)
    sm=pd.DataFrame(summaries)
    sm.to_csv(out/f"metrics_{a.shard_index:03d}.csv",index=False)
    if trade_parts:
      pd.concat(trade_parts,ignore_index=True).to_csv(out/f"trades_{a.shard_index:03d}.csv",index=False)
    manifest={"status":"completed","shard_index":a.shard_index,"shard_count":a.shard_count,
      "groups_expected":len(selected),"groups_completed":int(sm.group_id.nunique()),
      "metric_rows":len(sm),"expected_metric_rows":len(selected)*len(RRS)*len(PENDING_BARS)*len(CUTS),
      "symbols":list(SYMBOLS),"timeframes":list(TIMEFRAMES),"start":START,"end":END}
    if manifest["groups_completed"] != manifest["groups_expected"] or manifest["metric_rows"] != manifest["expected_metric_rows"]:
      raise RuntimeError(f"Incomplete shard output: {manifest}")
    (out/f"manifest_{a.shard_index:03d}.json").write_text(json.dumps(manifest,indent=2))
    print(f"SHARD_COMPLETE {json.dumps(manifest)}",flush=True)

if __name__=="__main__": main()
