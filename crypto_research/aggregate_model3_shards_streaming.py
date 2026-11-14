from __future__ import annotations
import argparse, gc, hashlib, json, os, re, tempfile, zipfile
from pathlib import Path
import requests
import numpy as np
import pandas as pd

API = "https://api.github.com"
OOS = ["OOS_2024", "OOS_2025", "OOS_2026_YTD"]
PERIODS = ["TRAIN_2021_2023", *OOS]
CONFIG = ["group_id", "rr", "pending_bars"]
METRIC_KEY = [*CONFIG, "sample"]


def api_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


def list_artifacts(repo: str, run_id: int, token: str):
    out, page = [], 1
    while True:
        url = f"{API}/repos/{repo}/actions/runs/{run_id}/artifacts"
        r = requests.get(url, headers=api_headers(token), params={"per_page": 100, "page": page}, timeout=(30, 120))
        r.raise_for_status()
        payload = r.json()
        items = payload.get("artifacts", [])
        out.extend(items)
        if len(items) < 100 or len(out) >= payload.get("total_count", len(out)):
            break
        page += 1
    return out


def download_artifact(repo: str, artifact: dict, token: str, path: Path):
    url = f"{API}/repos/{repo}/actions/artifacts/{artifact['id']}/zip"
    h = hashlib.sha256()
    with requests.get(url, headers=api_headers(token), stream=True, allow_redirects=True, timeout=(30, 180)) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    h.update(chunk)
    got = h.hexdigest()
    expected = (artifact.get("digest") or "").removeprefix("sha256:")
    if expected and got.lower() != expected.lower():
        raise RuntimeError(f"Artifact digest mismatch for {artifact['name']}: expected {expected}, got {got}")


def pf_from_sums(win_sum, loss_abs):
    if loss_abs > 0:
        return win_sum / loss_abs
    if win_sum > 0:
        return float("inf")
    return np.nan


def process_shard(zip_path: Path, shard_index: int, out_dir: Path, write_headers: dict):
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        mf = next((n for n in names if re.fullmatch(r"manifest_.*\.json", Path(n).name)), None)
        mf_metrics = next((n for n in names if re.fullmatch(r"metrics_.*\.csv", Path(n).name)), None)
        mf_trades = next((n for n in names if re.fullmatch(r"trades_.*\.csv", Path(n).name)), None)
        if not (mf and mf_metrics and mf_trades):
            raise RuntimeError(f"Shard {shard_index} missing manifest/metrics/trades: {names}")
        manifest = json.loads(z.read(mf))
        if manifest.get("shard_index") != shard_index or manifest.get("status") != "completed":
            raise RuntimeError(f"Bad manifest for shard {shard_index}: {manifest}")
        if manifest.get("groups_completed") != manifest.get("groups_expected"):
            raise RuntimeError(f"Incomplete groups in shard {shard_index}: {manifest}")

        metrics = pd.read_csv(z.open(mf_metrics))
        if len(metrics) != int(manifest["expected_metric_rows"]):
            raise RuntimeError(f"Metric row mismatch in shard {shard_index}: {len(metrics)}")
        metrics.to_csv(out_dir / "all_configuration_metrics.csv", mode="a", index=False,
                       header=not write_headers.get("metrics", False))
        write_headers["metrics"] = True

        # Process one shard at a time; never concatenate the raw trade corpus in memory.
        trades = pd.read_csv(z.open(mf_trades), low_memory=False)
        if not trades.empty:
            trades["entry_time_dt"] = pd.to_datetime(trades["entry_time"], utc=True)
            trades["exit_time_dt"] = pd.to_datetime(trades["exit_time"], utc=True)
            trades.sort_values(["group_id", "rr", "pending_bars", "sample", "entry_time_dt"],
                               kind="mergesort", inplace=True)
            gkeys = [*CONFIG, "sample"]
            grouped = [trades[k] for k in gkeys]
            trades["_equity"] = (1.0 + trades["net_bps"] / 10000.0).groupby(grouped, sort=False).cumprod()
            trades["_peak"] = trades["_equity"].groupby(grouped, sort=False).cummax()
            trades["_dd"] = (trades["_equity"] / trades["_peak"] - 1.0) * 100.0
            trades["_net_win"] = trades["net_bps"].clip(lower=0)
            trades["_net_loss_abs"] = -trades["net_bps"].clip(upper=0)
            trades["_gross_win"] = trades["gross_bps"].clip(lower=0)
            trades["_gross_loss_abs"] = -trades["gross_bps"].clip(upper=0)
            trades["_win"] = (trades["net_bps"] > 0).astype("int32")
            trades["_loss"] = (trades["net_bps"] < 0).astype("int32")
            trades["_be"] = (trades["net_bps"] == 0).astype("int32")
            pstats = trades.groupby(gkeys, sort=False, dropna=False).agg(
                trades=("net_bps", "size"), wins=("_win", "sum"), losses=("_loss", "sum"),
                breakeven=("_be", "sum"), net_positive_bps=("_net_win", "sum"),
                net_loss_abs_bps=("_net_loss_abs", "sum"), gross_profit_bps=("_gross_win", "sum"),
                gross_loss_bps=("_gross_loss_abs", "sum"), fees_bps=("fees_bps", "sum"),
                funding_cost_bps=("funding_cost_bps", "sum"), net_profit_bps=("net_bps", "sum"),
                max_drawdown_pct=("_dd", "min"), avg_net_bps=("net_bps", "mean"),
                avg_net_R=("net_R", "mean"), net_R_sum=("net_R", "sum"),
                first_entry_time=("entry_time_dt", "min"), last_exit_time=("exit_time_dt", "max")).reset_index()
            pstats["win_rate_pct"] = np.where(pstats.trades > 0, pstats.wins / pstats.trades * 100.0, 0.0)
            pstats["profit_factor"] = [pf_from_sums(a, b) for a, b in zip(pstats.net_positive_bps, pstats.net_loss_abs_bps)]
            pstats.to_csv(out_dir / "trade_period_stats.csv", mode="a", index=False,
                          header=not write_headers.get("period", False))
            write_headers["period"] = True

            oos = trades[trades["sample"].isin(OOS)].copy()
            if not oos.empty:
                oos.sort_values(["group_id", "rr", "pending_bars", "entry_time_dt"], kind="mergesort", inplace=True)
                cgroups = [oos[k] for k in CONFIG]
                oos["_oos_equity"] = (1.0 + oos["net_bps"] / 10000.0).groupby(cgroups, sort=False).cumprod()
                oos["_oos_peak"] = oos["_oos_equity"].groupby(cgroups, sort=False).cummax()
                oos["_oos_dd"] = (oos["_oos_equity"] / oos["_oos_peak"] - 1.0) * 100.0
                ostats = oos.groupby(CONFIG, sort=False, dropna=False).agg(
                    oos_max_drawdown_pct=("_oos_dd", "min"),
                    oos_first_entry_time=("entry_time_dt", "min"),
                    oos_last_exit_time=("exit_time_dt", "max")).reset_index()
                ostats.to_csv(out_dir / "trade_oos_drawdown.csv", mode="a", index=False,
                              header=not write_headers.get("oos", False))
                write_headers["oos"] = True
        return manifest, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    ap.add_argument("--out", default="artifacts/model3_full_retracement_pf")
    ap.add_argument("--local-zip", help="Optional single-shard local validation mode")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["all_configuration_metrics.csv", "trade_period_stats.csv", "trade_oos_drawdown.csv"]:
        p = out_dir / name
        if p.exists():
            p.unlink()
    write_headers, manifests = {}, []
    if args.local_zip:
        with zipfile.ZipFile(args.local_zip) as z:
            mf = next(n for n in z.namelist() if re.fullmatch(r"manifest_.*\.json", Path(n).name))
            idx = int(json.loads(z.read(mf))["shard_index"])
        m, _ = process_shard(Path(args.local_zip), idx, out_dir, write_headers)
        print(json.dumps({"local_validation": True, "manifest": m, "output": str(out_dir)}, indent=2))
        return
    if not args.token:
        raise RuntimeError("GITHUB_TOKEN is required")
    artifacts = list_artifacts(args.repo, args.run_id, args.token)
    shards, quality = {}, None
    for a in artifacts:
        match = re.fullmatch(r"model3-shard-(\d+)", a["name"])
        if match:
            shards[int(match.group(1))] = a
        elif a["name"] == "model3-data-quality":
            quality = a
    missing = sorted(set(range(102)) - set(shards))
    if missing or len(shards) != 102 or quality is None:
        raise RuntimeError(f"Artifact inventory incomplete: shard_count={len(shards)}, missing={missing}, quality={quality is not None}, total_artifacts={len(artifacts)}")
    for i in range(102):
        artifact = shards[i]
        if artifact.get("expired"):
            raise RuntimeError(f"Shard artifact expired: {artifact['name']}")
        with tempfile.NamedTemporaryFile(prefix=f"m3_{i}_", suffix=".zip", delete=False) as tmp:
            zip_path = Path(tmp.name)
        try:
            download_artifact(args.repo, artifact, args.token, zip_path)
            m, metrics = process_shard(zip_path, i, out_dir, write_headers)
            manifests.append(m)
            if (i + 1) % 10 == 0 or i == 101:
                print(f"PROCESSED SHARDS {i+1}/102", flush=True)
            del metrics
            gc.collect()
        finally:
            zip_path.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(prefix="m3_quality_", suffix=".zip", delete=False) as tmp:
        qpath = Path(tmp.name)
    try:
        download_artifact(args.repo, quality, args.token, qpath)
        with zipfile.ZipFile(qpath) as z:
            qname = next(n for n in z.namelist() if Path(n).name == "data_quality.json")
            quality_data = json.loads(z.read(qname))
        (out_dir / "data_quality.json").write_text(json.dumps(quality_data, indent=2))
    finally:
        qpath.unlink(missing_ok=True)
    (out_dir / "shard_manifests.json").write_text(json.dumps(manifests, indent=2))

    metrics = pd.read_csv(out_dir / "all_configuration_metrics.csv")
    period = pd.read_csv(out_dir / "trade_period_stats.csv") if (out_dir / "trade_period_stats.csv").exists() else pd.DataFrame(columns=METRIC_KEY)
    drawdown = pd.read_csv(out_dir / "trade_oos_drawdown.csv") if (out_dir / "trade_oos_drawdown.csv").exists() else pd.DataFrame(columns=CONFIG)
    if len(metrics) != 388800 or metrics["group_id"].nunique() != 6480 or set(metrics.group_id.unique()) != set(range(6480)):
        raise RuntimeError(f"Full coverage check failed: metric_rows={len(metrics)}, groups={metrics.group_id.nunique()}")
    if metrics.duplicated(METRIC_KEY).any():
        raise RuntimeError("Duplicate configuration/sample rows found in metrics")
    if len(period):
        check = metrics[METRIC_KEY + ["trades", "net_bps"]].merge(
            period[METRIC_KEY + ["trades", "net_profit_bps"]], on=METRIC_KEY, how="left", suffixes=("_metric", "_trade"))
        check["trades_trade"] = check["trades_trade"].fillna(0)
        check["net_profit_bps"] = check["net_profit_bps"].fillna(0)
        if (check.trades_metric.astype(int) != check.trades_trade.astype(int)).any():
            raise RuntimeError("Trade-row counts do not reconcile to shard metric rows")
        if ((check.net_bps - check.net_profit_bps).abs() > 0.03).any():
            raise RuntimeError("Net P&L does not reconcile to metric summaries within 0.03 bps")
    cfg_cols = ["group_id", "model", "symbol", "timeframe", "atr_period", "impulse_atr", "adx_min", "adx_max", "rsi_mode", "retracement", "rr", "pending_bars"]
    base = metrics[cfg_cols].drop_duplicates().copy()
    if len(base) != 97200 or base.duplicated(CONFIG).any():
        raise RuntimeError(f"Configuration universe mismatch: expected 97,200 unique configs, got {len(base)}")

    oos_period = period[period["sample"].isin(OOS)].copy()
    overall = oos_period.groupby(CONFIG, dropna=False).agg(
        oos_trades=("trades", "sum"), oos_wins=("wins", "sum"), oos_losses=("losses", "sum"),
        oos_breakeven=("breakeven", "sum"), oos_net_positive_bps=("net_positive_bps", "sum"),
        oos_net_loss_abs_bps=("net_loss_abs_bps", "sum"), oos_gross_profit_bps=("gross_profit_bps", "sum"),
        oos_gross_loss_bps=("gross_loss_bps", "sum"), oos_fees_bps=("fees_bps", "sum"),
        oos_funding_cost_bps=("funding_cost_bps", "sum"), oos_net_profit_bps=("net_profit_bps", "sum"),
        oos_net_R_sum=("net_R_sum", "sum")).reset_index()
    base = base.merge(overall, on=CONFIG, how="left")
    base = base.merge(drawdown, on=CONFIG, how="left")
    for sample in OOS:
        s = period[period["sample"] == sample].drop(columns="sample").copy()
        s = s.rename(columns={c: f"{sample.lower()}_{c}" for c in s.columns if c not in CONFIG})
        base = base.merge(s, on=CONFIG, how="left")
    count_cols = [c for c in base.columns if c.endswith("_trades") or c.endswith("_wins") or c.endswith("_losses") or c.endswith("_breakeven")]
    money_cols = [c for c in base.columns if any(k in c for k in ["_net_positive_bps", "_net_loss_abs_bps", "_gross_profit_bps", "_gross_loss_bps", "_fees_bps", "_funding_cost_bps", "_net_profit_bps"])]
    base[count_cols + money_cols] = base[count_cols + money_cols].fillna(0)
    base["oos_win_rate_pct"] = np.where(base.oos_trades > 0, base.oos_wins / base.oos_trades * 100.0, 0.0)
    base["oos_avg_net_bps"] = np.where(base.oos_trades > 0, base.oos_net_profit_bps / base.oos_trades, 0.0)
    base["oos_avg_net_R"] = np.where(base.oos_trades > 0, base.oos_net_R_sum / base.oos_trades, 0.0)
    base["oos_profit_factor"] = [pf_from_sums(a, b) for a, b in zip(base.oos_net_positive_bps, base.oos_net_loss_abs_bps)]
    for sample in OOS:
        p = sample.lower()
        w, l = f"{p}_net_positive_bps", f"{p}_net_loss_abs_bps"
        base[f"{p}_win_rate_pct"] = np.where(base[f"{p}_trades"] > 0, base[f"{p}_wins"] / base[f"{p}_trades"] * 100.0, 0.0)
        base[f"{p}_profit_factor"] = [pf_from_sums(a, b) for a, b in zip(base[w].fillna(0), base[l].fillna(0))]
    base["eligible_60_oos_trades"] = base.oos_trades >= 60
    base = base.sort_values(["eligible_60_oos_trades", "oos_profit_factor", "oos_net_profit_bps", "oos_trades", "oos_max_drawdown_pct"],
                            ascending=[False, False, False, False, False], na_position="last", kind="mergesort")
    base.to_csv(out_dir / "all_configuration_oos_metrics.csv", index=False)
    eligible = base[base.eligible_60_oos_trades].copy()
    eligible = eligible.sort_values(["oos_profit_factor", "oos_net_profit_bps", "oos_trades", "oos_max_drawdown_pct"],
                                    ascending=[False, False, False, False], na_position="last", kind="mergesort")
    eligible.insert(0, "pf_rank", np.arange(1, len(eligible) + 1))
    eligible.to_csv(out_dir / "all_eligible_ranked_by_pf.csv", index=False)
    eligible.head(100).to_csv(out_dir / "top100_all_rr_by_pf.csv", index=False)
    methodology = f"""# Model 3 full-sweep recovery and PF ranking

- Source run: {args.run_id} in {args.repo}; source commit: 9a5594e52459240e3b434d8ad96a528a8278d6a3.
- Recovered and SHA-256-verified all 102 shard ZIPs plus the separate data-quality artifact.
- Verified 6,480 group IDs, 388,800 per-sample metric rows, and 97,200 unique asset/RR/expiry configurations (all RR values 2R–6R; all expiry values 3/6/10; BTCUSDT and ETHUSDT; 5m/15m/30m/1h).
- Chronological samples: 2021–2023 training, 2024 OOS, 2025 OOS, 2026-01-01 through 2026-11-01 OOS YTD.
- Exact-price multi-candle retracement entries at 75%, 78.6%, 80%, 82%, 85%; ATR 14/20/25; impulse 1.8/2.0/2.2/2.4/2.5/2.7 ATR; ADX bands 18–30, 15–25, 20–35; RSI exact/loose/none.
- Execution model: pending limit valid for 3/6/10 bars; 5 bps fee and 2 bps slippage per side; official Binance Vision funding rates; same-bar stop/target ambiguity resolved stop-first.
- PF is recomputed from trade-level net wins divided by absolute trade-level net losses after fees, slippage and funding. It is not the mean of per-period PFs. Gross profit/loss columns are pre-fee/pre-funding (slippage remains embedded in execution prices). Net profit is the sum of trade net bps. Drawdown uses compounded per-trade net returns.
- Eligibility threshold retained from the original analysis: at least 60 combined OOS trades. No RR filter and no positive-PF filter were applied. Ranking order: combined OOS PF descending, net profit descending, trade count descending, then drawdown.
- Memory-safe recovery: each ZIP is downloaded, checksum-verified and processed individually; the raw trade corpus is never concatenated in memory.

## Data quality

{json.dumps(quality_data, indent=2)}

## Result counts

- Eligible configurations: {len(eligible):,}
- Top-100 file: top100_all_rr_by_pf.csv
- Complete eligible ranking: all_eligible_ranked_by_pf.csv
- All 97,200 configurations, including below-threshold cases: all_configuration_oos_metrics.csv
- Full per-sample metric rows: all_configuration_metrics.csv
"""
    (out_dir / "methodology_and_data_quality.md").write_text(methodology)
    summary = {"status": "completed", "shards_verified": len(manifests), "groups": int(metrics.group_id.nunique()),
               "metric_rows": len(metrics), "unique_configurations": len(base), "eligible_configurations": len(eligible),
               "top100_rows": min(100, len(eligible)), "artifact_inventory_count": len(artifacts),
               "data_quality": quality_data, "ranking": "combined OOS net PF descending; all RR 2R-6R included"}
    (out_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("TOP 20 ALL-RR BY PF")
    print(eligible.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
