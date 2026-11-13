from pathlib import Path
import json
import pandas as pd
from .sweep_m3_full_retracement import CUTS
def main():
    root=Path("artifacts/shards")
    manifests=sorted(root.rglob("manifest_*.json"))
    expected_shards=102
    ms=[json.loads(p.read_text()) for p in manifests]
    ids={m["shard_index"] for m in ms}
    if len(ms)!=expected_shards or ids!=set(range(expected_shards)):
        raise RuntimeError(f"Missing/duplicate shards: got {len(ms)} manifests; ids={sorted(ids)}")
    for m in ms:
        if m["status"]!="completed" or m["groups_completed"]!=m["groups_expected"] or m["metric_rows"]!=m["expected_metric_rows"]:
            raise RuntimeError(f"Bad shard manifest: {m}")
    metric_files=sorted(root.rglob("metrics_*.csv"))
    trade_files=sorted(root.rglob("trades_*.csv"))
    sm=pd.concat([pd.read_csv(p) for p in metric_files],ignore_index=True)
    if sm.group_id.nunique()!=6480 or len(sm)!=6480*5*3*4:
        raise RuntimeError(f"Coverage failure: groups={sm.group_id.nunique()} rows={len(sm)}")
    if set(sm.group_id.unique())!=set(range(6480)):
        raise RuntimeError("Missing group IDs")
    out=Path("artifacts/model3_full_retracement"); out.mkdir(parents=True,exist_ok=True)
    sm.to_csv(out/"all_configuration_metrics.csv",index=False)
    if trade_files:
        pd.concat([pd.read_csv(p) for p in trade_files],ignore_index=True).to_csv(out/"all_trades.csv",index=False)
    fwd=sm[sm.sample.isin(["OOS_2024","OOS_2025","OOS_2026_YTD"])].copy()
    keys=["model","symbol","timeframe","atr_period","impulse_atr","adx_min","adx_max","rsi_mode","retracement","rr","pending_bars"]
    agg=fwd.groupby(keys,dropna=False).agg(trades=("trades","sum"),net_bps=("net_bps","sum"),
      win_rate_pct=("win_rate_pct","mean"),pf=("profit_factor","mean"),
      years_positive=("net_bps",lambda x:int((x>0).sum()))).reset_index()
    agg=agg[agg.trades>=60].sort_values(["years_positive","net_bps"],ascending=False)
    agg.to_csv(out/"forward_robustness_ranking.csv",index=False)
    level=fwd.groupby(["retracement","rr","pending_bars"]).agg(trades=("trades","sum"),
      net_bps=("net_bps","sum"),mean_win_rate=("win_rate_pct","mean"),
      mean_pf=("profit_factor","mean")).reset_index()
    level.to_csv(out/"retracement_rr_level_comparison.csv",index=False)
    manifest={"status":"completed","verified_shards":len(ms),"groups":int(sm.group_id.nunique()),
      "metric_rows":len(sm),"symbols":["BTCUSDT","ETHUSDT"],"periods":[x[0] for x in CUTS],
      "matrix":{"timeframes":4,"atr_periods":3,"impulse_strengths":6,"adx_bands":3,
        "rsi_modes":3,"retracements":5,"rr":5,"pending_expiries":3},
      "asset_expiry_groups":6480,"sample_evaluations":len(sm),
      "fees_bps_per_side":5,"slippage_bps_per_side":2}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("FULL MATRIX VERIFIED",json.dumps(manifest),flush=True)
    print("TOP 30 FORWARD CANDIDATES")
    print(agg.head(30).to_string(index=False))
    print("RETRACEMENT/RR/EXPIRY SUMMARY")
    print(level.to_string(index=False))
if __name__=="__main__": main()
