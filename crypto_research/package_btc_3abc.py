from pathlib import Path
import json, shutil, zipfile
import pandas as pd

ROOT=Path(".")
REPORT=Path("inputs/report")
POOL=Path("inputs/pool")
OUT=Path("dist")
BUNDLE=OUT/"BTC_Model_3ABC_Backtested_Results_and_Exact_Code.zip"

def main():
    OUT.mkdir(exist_ok=True)
    # gh run download extracts artifact contents directly into these folders.
    rd=REPORT
    pdx=POOL
    config_path=rd/"ASSET_ORIENTED_MODEL3_CONFIGS.json"
    configs=json.loads(config_path.read_text())
    baseline=configs["btc_frozen_baseline_46"]
    ids=[]
    for family, rows in baseline.items():
        if family not in ("3A","3B","3C"): continue
        for row in rows:
            row["family"]=family
            ids.append(row["candidate_id"])
    if len(ids)!=46 or len(set(ids))!=46:
        raise RuntimeError(f"Expected exactly 46 unique frozen BTC 3A/3B/3C configurations, got {len(ids)}")
    metrics=pd.read_csv(rd/"TOP2K_ALL_ASSET_ALL_CONFIG_METRICS.csv")
    btc=metrics[(metrics.symbol=="BTCUSDT") & metrics.candidate_id.astype(str).isin(ids)].copy()
    if btc.candidate_id.nunique()!=46 or len(btc)!=46*4:
        raise RuntimeError(f"BTC configuration metrics incomplete: configs={btc.candidate_id.nunique()}, rows={len(btc)}")
    family_map={r["candidate_id"]:fam for fam,rows in baseline.items() for r in rows}
    btc["family"]=btc.candidate_id.map(family_map)
    if btc.family.isna().any(): raise RuntimeError("Some baseline configs have no 3A/3B/3C family")
    btc.to_csv(OUT/"BTC_3ABC_46_CONFIGS_ALL_PERIODS.csv",index=False)
    oos=btc[btc.period.str.startswith("OOS_")].copy()
    oos.to_csv(OUT/"BTC_3ABC_46_CONFIGS_OOS_BY_YEAR.csv",index=False)
    # Aggregate each individual configuration across all OOS years using net trade outcomes.
    indiv=oos.groupby(["family","candidate_id","timeframe","atr_period","impulse_atr","adx_min","adx_max","rsi_mode","retracement","rr","pending_bars"],dropna=False).agg(
        trades=("trades","sum"),wins=("wins","sum"),losses=("losses","sum"),
        net_winning_bps=("net_winning_bps","sum"),net_losing_bps=("net_losing_bps","sum"),
        net_bps=("net_bps","sum"),worst_period_drawdown_pct=("max_drawdown_pct","min"),
        long_trades=("long_trades","sum"),short_trades=("short_trades","sum")).reset_index()
    indiv["win_rate_pct"]=100*indiv.wins/indiv.trades
    indiv["combined_oos_pf"]=indiv.net_winning_bps/indiv.net_losing_bps.replace({0:float("nan")})
    indiv=indiv.sort_values(["family","combined_oos_pf","net_bps"],ascending=[True,False,False])
    indiv.to_csv(OUT/"BTC_3ABC_46_CONFIGS_COMBINED_OOS.csv",index=False)
    fam=oos.groupby(["family","period"],dropna=False).agg(
        configurations=("candidate_id","nunique"),trades=("trades","sum"),wins=("wins","sum"),losses=("losses","sum"),
        net_winning_bps=("net_winning_bps","sum"),net_losing_bps=("net_losing_bps","sum"),
        net_bps=("net_bps","sum"),positive_configs=("net_bps",lambda x:int((x>0).sum())),
        worst_config_drawdown_pct=("max_drawdown_pct","min"),median_config_drawdown_pct=("max_drawdown_pct","median")).reset_index()
    fam["win_rate_pct"]=100*fam.wins/fam.trades
    fam["combined_pf"]=fam.net_winning_bps/fam.net_losing_bps.replace({0:float("nan")})
    fam.to_csv(OUT/"BTC_3ABC_FAMILY_OOS_BY_YEAR.csv",index=False)
    allfam=oos.groupby("family",dropna=False).agg(
        configurations=("candidate_id","nunique"),trades=("trades","sum"),wins=("wins","sum"),losses=("losses","sum"),
        net_winning_bps=("net_winning_bps","sum"),net_losing_bps=("net_losing_bps","sum"),
        net_bps=("net_bps","sum"),positive_configs=("net_bps",lambda x:int((x>0).sum()))).reset_index()
    allfam["win_rate_pct"]=100*allfam.wins/allfam.trades
    allfam["combined_oos_pf"]=allfam.net_winning_bps/allfam.net_losing_bps.replace({0:float("nan")})
    allfam.to_csv(OUT/"BTC_3ABC_FAMILY_COMBINED_OOS.csv",index=False)
    # Preserve full family configuration definitions and the exact frozen BTC config.
    (OUT/"BTC_3ABC_FROZEN_CONFIGURATIONS.json").write_text(json.dumps(baseline,indent=2))
    # Pull pool and manifest from the actual candidate-pool artifact.
    for name in ["model3_top2000_unique_candidate_pool.csv","top2k_pool_manifest.json"]:
        found=list(pdx.rglob(name))
        if found: shutil.copy2(found[0],OUT/name)
    # Add source files verbatim from the exact source commit used for the top-2k transfer result.
    source=OUT/"exact_source_code"
    for rel in [
        "crypto_research/__init__.py",
        "crypto_research/data.py",
        "crypto_research/sweep_m3_strong_misjudge.py",
        "crypto_research/sweep_m3_full_retracement.py",
        "crypto_research/prepare_model3_top2k_pool.py",
        "crypto_research/model3_top2k_transfer.py",
        "requirements.txt",
    ]:
        p=ROOT/rel
        if not p.exists(): raise RuntimeError(f"Exact source file missing from checkout: {rel}")
        dst=source/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dst)
    # Include BTC's audited data-quality evidence from the final run.
    coverage=pd.read_csv(rd/"DATA_AND_FUNDING_COVERAGE.csv")
    coverage[coverage.symbol=="BTCUSDT"].to_csv(OUT/"BTC_DATA_AND_FUNDING_COVERAGE.csv",index=False)
    (OUT/"SOURCE_AND_METHOD.md").write_text("""# BTC Model 3A / 3B / 3C — backtest bundle

This bundle preserves the BTC frozen baseline of 46 independent configurations: 21 in 3A, 8 in 3B and 17 in 3C. The exact parameter definitions and source files are included.

## Backtest periods
- Development: 2021-01-01 to 2023-12-31
- OOS 2024: 2024-01-01 to 2024-12-31
- OOS 2025: 2025-01-01 to 2025-12-31
- OOS 2026 YTD: 2026-01-01 to 2026-10-31

## Execution assumptions
Official Binance USD-M 1-minute candles, resampled to 1h for the frozen BTC baseline; official Binance Vision funding rates; 5 bps fee and 2 bps slippage per side; exact-price multi-candle retracement limit entry; configured 3-bar pending expiry; stop-first same-bar ambiguity; 24-bar maximum holding horizon.

## Included result files
- BTC_3ABC_FAMILY_OOS_BY_YEAR.csv: family-level combined PF by OOS year.
- BTC_3ABC_FAMILY_COMBINED_OOS.csv: each family combined over all OOS years.
- BTC_3ABC_46_CONFIGS_COMBINED_OOS.csv: individual configuration PF across all OOS years.
- BTC_3ABC_46_CONFIGS_OOS_BY_YEAR.csv: individual configuration results by OOS year.
- BTC_3ABC_46_CONFIGS_ALL_PERIODS.csv: individual configuration results including development.
- BTC_3ABC_FROZEN_CONFIGURATIONS.json: exact 46 parameter configurations.

Combined PF is computed from pooled net winning bps divided by pooled absolute net losing bps. It is not the arithmetic mean of yearly PFs. Family-level totals are research aggregates, not a live portfolio simulation: they do not model shared capital, leverage or overlapping position constraints.

The source files are copied verbatim from the research commit used for the BTC frozen-baseline transfer evaluation. The BTC family definitions are provided in the included frozen configuration JSON.
""")
    # Make one easy-to-open archive with both results and exact source code.
    with zipfile.ZipFile(BUNDLE,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in OUT.rglob("*"):
            if p.is_file() and p!=BUNDLE:
                z.write(p,p.relative_to(OUT))
    print(f"Created {BUNDLE} ({BUNDLE.stat().st_size} bytes)")
    print(allfam.to_string(index=False))
    print(f"Configurations: {len(ids)}; metrics: {len(btc)}; OOS rows: {len(oos)}")

if __name__=="__main__": main()
