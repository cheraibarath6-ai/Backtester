from pathlib import Path
import json
import numpy as np
import pandas as pd
import shutil
from .sweep_m3_full_retracement import CUTS, SYMBOLS, RRS, PENDING_BARS

SHARD_COUNT=25
OOS=["OOS_2024","OOS_2025","OOS_2026_YTD"]
KEY=["symbol","group_id","rr","pending_bars"]

def pf(wins,losses):
    return wins/losses if losses>0 else (float("inf") if wins>0 else 0.)

def main():
    root=Path("artifacts/shards")
    out=Path("artifacts/model3_full_universe_asset_specific")
    out.mkdir(parents=True,exist_ok=True)
    manifests=sorted(root.rglob("manifest_*.json"))
    expected={(s,i) for s in SYMBOLS for i in range(SHARD_COUNT)}
    got=set()
    for p in manifests:
        m=json.loads(p.read_text())
        k=(m.get("symbol"),m.get("shard_index"))
        if k in got: raise RuntimeError(f"Duplicate shard manifest {k}")
        got.add(k)
        if m.get("status")!="completed" or m.get("groups_completed")!=m.get("groups_expected") or m.get("metric_rows")!=m.get("expected_metric_rows"):
            raise RuntimeError(f"Incomplete shard: {m}")
    if got!=expected:
        raise RuntimeError(f"Missing/extra shards: missing={sorted(expected-got)} extra={sorted(got-expected)}")

    files=sorted(root.rglob("metrics_*.csv"))
    if len(files)!=len(expected): raise RuntimeError(f"Expected {len(expected)} metrics files, found {len(files)}")
    metrics=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    base_sets=4*3*6*3*3*5
    candidates_per_asset=base_sets*len(RRS)*len(PENDING_BARS)
    rows_per_asset=candidates_per_asset*len(CUTS)
    expected_rows=len(SYMBOLS)*rows_per_asset
    if len(metrics)!=expected_rows:
        raise RuntimeError(f"Metric row mismatch: expected {expected_rows}, got {len(metrics)}")
    if metrics.duplicated(KEY+["sample"]).any():
        raise RuntimeError("Duplicate asset/configuration/period rows")
    for s in SYMBOLS:
        m=metrics[metrics.symbol==s]
        if m.group_id.nunique()!=base_sets or m.groupby("sample").group_id.nunique().ne(base_sets).any():
            raise RuntimeError(f"{s}: expected all {base_sets} base parameter groups in every sample")
        if m.groupby(KEY).size().ne(len(CUTS)).any():
            raise RuntimeError(f"{s}: incomplete period coverage for one or more candidates")
        if m[KEY].drop_duplicates().shape[0]!=candidates_per_asset:
            raise RuntimeError(f"{s}: expected {candidates_per_asset} unique configurations")

    metrics.to_csv(out/"ALL_ASSET_ALL_CONFIGURATION_METRICS.csv",index=False)
    train=metrics[metrics["sample"]=="TRAIN_2021_2023"].copy()
    train["candidate_id"]=train.apply(lambda r:f"{r['model']}|RR{r['rr']:g}|EXP{int(r['pending_bars'])}",axis=1)
    pf_component=np.log(np.clip(pd.to_numeric(train.profit_factor,errors="coerce").replace([np.inf,-np.inf],5).fillna(0.01),0.01,5.0))
    train["selection_score"]=(0.55*pf_component+
        0.15*np.log1p(train.trades.clip(lower=0))+
        0.20*np.clip(train.net_bps/10000.0,-2,2)-
        0.10*(train.max_drawdown_pct.abs()/100.0))
    train["development_qualified"]=(train.trades>=100)&(train.net_bps>0)&(train.max_drawdown_pct.abs()<=35)&(train.profit_factor>=1.5)
    train["development_positive"]=(train.trades>=100)&(train.net_bps>0)&(train.max_drawdown_pct.abs()<=35)
    train.to_csv(out/"ALL_ASSET_CONFIGURATION_DEVELOPMENT_RANKING.csv",index=False)

    chosen=[]
    for s in SYMBOLS:
        a=train[train.symbol==s].copy()
        tiers=[
            a[(a.trades>=100)&(a.net_bps>0)&(a.max_drawdown_pct.abs()<=35)],
            a[a.trades>=100],
            a[a.trades>=50],
            a
        ]
        pool=None
        for tier in tiers:
            if len(tier)>=2:
                pool=tier
                break
        if pool is None or pool.empty:
            raise RuntimeError(f"{s}: no candidates available for top-two selection")
        top=pool.sort_values(["selection_score","profit_factor","net_bps","trades"],
                             ascending=[False,False,False,False],na_position="last").head(2).copy()
        top["selection_rank"]=[1,2]
        top["selection_tier"]=("POSITIVE_NET_100_TRADES_DD35" if len(tiers[0])>=2 and set(top.candidate_id)<=set(tiers[0].candidate_id)
                               else "FALLBACK_BEST_AVAILABLE")
        top["development_pf_ge_1_5"]=top.profit_factor>=1.5
        chosen.append(top)
    selected=pd.concat(chosen,ignore_index=True)
    if len(selected)!=20 or selected.groupby("symbol").size().ne(2).any():
        raise RuntimeError("Top-two selection did not produce exactly two independent configs per asset")
    selected.to_csv(out/"ASSET_TOP2_SELECTED_CONFIGS.csv",index=False)

    selected_keys=selected[KEY+["candidate_id","selection_rank","selection_tier","selection_score"]]
    selected_metrics=metrics.merge(selected_keys,on=KEY,how="inner",validate="many_to_one")
    selected_metrics.to_csv(out/"ASSET_TOP2_ALL_PERIOD_METRICS.csv",index=False)
    oos=selected_metrics[selected_metrics["sample"].isin(OOS)].copy()
    individual=oos.groupby(KEY+["candidate_id","selection_rank","selection_tier"],dropna=False).agg(
        trades=("trades","sum"),wins=("wins","sum"),losses=("losses","sum"),
        net_winning_bps=("net_winning_bps","sum"),net_losing_bps=("net_losing_bps","sum"),
        net_bps=("net_bps","sum"),worst_period_drawdown_pct=("max_drawdown_pct","min"),
        positive_years=("net_bps",lambda x:int((x>0).sum()))).reset_index()
    individual["win_rate_pct"]=np.where(individual.trades>0,100*individual.wins/individual.trades,0)
    individual["combined_oos_pf"]=[pf(a,b) for a,b in zip(individual.net_winning_bps,individual.net_losing_bps)]
    individual.to_csv(out/"ASSET_TOP2_INDIVIDUAL_OOS_SUMMARY.csv",index=False)
    yearly=oos.groupby(KEY+["candidate_id","selection_rank","sample"],dropna=False).agg(
        trades=("trades","sum"),wins=("wins","sum"),losses=("losses","sum"),
        net_winning_bps=("net_winning_bps","sum"),net_losing_bps=("net_losing_bps","sum"),
        net_bps=("net_bps","sum"),max_drawdown_pct=("max_drawdown_pct","min")).reset_index()
    yearly["win_rate_pct"]=np.where(yearly.trades>0,100*yearly.wins/yearly.trades,0)
    yearly["profit_factor"]=[pf(a,b) for a,b in zip(yearly.net_winning_bps,yearly.net_losing_bps)]
    yearly.to_csv(out/"ASSET_TOP2_INDIVIDUAL_OOS_BY_YEAR.csv",index=False)

    pair=yearly.groupby(["symbol","sample"],dropna=False).agg(
        configurations=("candidate_id","nunique"),trades=("trades","sum"),wins=("wins","sum"),losses=("losses","sum"),
        net_winning_bps=("net_winning_bps","sum"),net_losing_bps=("net_losing_bps","sum"),
        net_bps=("net_bps","sum"),worst_configuration_drawdown_pct=("max_drawdown_pct","min")).reset_index()
    pair["win_rate_pct"]=np.where(pair.trades>0,100*pair.wins/pair.trades,0)
    pair["combined_pf"]=[pf(a,b) for a,b in zip(pair.net_winning_bps,pair.net_losing_bps)]
    pair.to_csv(out/"ASSET_TOP2_COMBINED_PF_BY_YEAR.csv",index=False)
    overall=oos.groupby("symbol",dropna=False).agg(
        configurations=("candidate_id","nunique"),trades=("trades","sum"),wins=("wins","sum"),losses=("losses","sum"),
        net_winning_bps=("net_winning_bps","sum"),net_losing_bps=("net_losing_bps","sum"),
        net_bps=("net_bps","sum"),worst_configuration_drawdown_pct=("max_drawdown_pct","min")).reset_index()
    overall["win_rate_pct"]=np.where(overall.trades>0,100*overall.wins/overall.trades,0)
    overall["combined_oos_pf"]=[pf(a,b) for a,b in zip(overall.net_winning_bps,overall.net_losing_bps)]
    overall.to_csv(out/"ASSET_TOP2_COMBINED_PF_OVERALL.csv",index=False)

    quality=Path("artifacts/preflight/data_quality.json")
    if quality.exists(): shutil.copy2(quality,out/"DATA_AND_FUNDING_QUALITY.json")
    manifest={"status":"completed","assets":list(SYMBOLS),"base_parameter_sets_per_asset":base_sets,
      "candidate_configurations_per_asset":candidates_per_asset,
      "total_asset_specific_candidate_configurations":candidates_per_asset*len(SYMBOLS),
      "metric_rows":len(metrics),"selected_configurations":len(selected),
      "selected_per_asset":2,"development_period":"2021-2023",
      "oos_periods":OOS,"fees_bps_per_side":5,"slippage_bps_per_side":2,
      "funding":"Official Binance Vision USD-M fundingRate; preflight requires no detected missing settlement gaps",
      "selection":"Each asset independently ranked on development data; no cross-asset transfer, ensemble, voting or confirmation",
      "verified_shards":len(manifests),"shards_per_asset":SHARD_COUNT,
      "method":"Exact-price multi-candle retracement limit entry; 75/78.6/80/82/85% retracement; RR 2-6; pending 3/6/10 bars; 5m/15m/30m/1h"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("FULL ASSET-SPECIFIC MATRIX VERIFIED",json.dumps(manifest),flush=True)
    print("TOP TWO PER ASSET WITH DEVELOPMENT PF")
    print(selected[["symbol","selection_rank","candidate_id","trades","profit_factor","net_bps","max_drawdown_pct","selection_tier"]].to_string(index=False))
    print("COMBINED PF BY ASSET")
    print(overall[["symbol","trades","combined_oos_pf","net_bps","win_rate_pct"]].to_string(index=False))

if __name__=="__main__": main()
