# Model 3 Top-2,000 Cross-Asset Transfer Research

This research stage takes the exact 2,000 highest-ranked profitable rows from the recovered 97,200-configuration Model 3 sweep, collapses duplicate parameter sets shared by BTC and ETH, and tests the resulting 1,853 unique configurations on all ten Binance USD-M perpetual symbols:

`BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, LINKUSDT, LTCUSDT, AVAXUSDT`.

## Important preservation rule

The existing BTC Model 3A/3B/3C configuration file is copied verbatim as `model3_btc_frozen_baseline_46.json` (21 + 8 + 17 = 46 configurations). This research workflow does not edit the currently running BTC configuration or its live script. Top-2,000 candidates are evaluated separately for BTC as a research comparison; BTC's active baseline remains the frozen 46 configurations.

## Candidate pool

- Original ranking: top 2,000 eligible rows from `all_eligible_ranked_by_pf.csv`, ranked by combined 2024–2026 YTD OOS PF with trade count and drawdown tie-breaks.
- Source assets: BTCUSDT and ETHUSDT only.
- Source row count: 2,000.
- Unique parameter sets after deduplication across source assets: 1,853.
- Candidate dimensions: 15m/30m/1h, ATR 14/20/25, impulse 1.8–2.7 ATR, ADX bands 15–25/18–30/20–35, RSI exact/loose/none, retracement 75–85%, RR 2R–6R, pending expiry 3/6/10 bars.
- Broad archetypes 3A/3B/3C are retained; candidates outside those established archetypes remain in 3X rather than being discarded.

## Per-asset selection and test design

1. Download official Binance USD-M 1-minute klines for each asset.
2. Obtain funding rates only from the official Binance Futures funding-rate API. No third-party funding fallback and no zero-funding assumption.
3. Build only complete 15m/30m/1h bars; report actual listing/data start and coverage.
4. Replay every unique candidate independently with exact-price multi-candle retracement limit entries, 5 bps fee and 2 bps slippage per side, official funding, stop-first same-bar ambiguity, and the original timeframe-specific maximum holding horizon.
5. Select up to 15 configurations per broad family and asset using 2021–2023 development metrics only, with at least 60 trades, PF >= 1.0 and positive net bps. If a family has no qualifying candidate, it remains empty; no configuration is forced into service.
6. Report 2024, 2025 and 2026 YTD as evaluation periods. They do not drive asset-specific candidate selection.

## Interpretation caveat

The source top-2,000 pool itself was ranked using BTC/ETH 2024–2026 YTD OOS results. Applying that preselected pool to other assets is useful transfer research, but not a fully independent blind experiment because crypto assets are correlated and the source pool was selected on those same calendar regimes. The per-asset selection is restricted to 2021–2023 development metrics, and later periods are shown separately. A fresh forward paper-trading period is still required before live use.

## Output artifacts

- `TOP2K_ALL_ASSET_ALL_CONFIG_METRICS.csv`: all candidate metrics by asset and period.
- `TOP2K_ASSET_OOS_SUMMARY.csv`: per-asset OOS diagnostics.
- `ASSET_ORIENTED_SELECTED_CONFIGS.csv`: development-selected asset candidates.
- `ASSET_ORIENTED_MODEL3_CONFIGS.json`: selected asset candidates plus the unchanged BTC 46-config baseline.
- Per-asset data quality, funding coverage and run summaries.

All configuration streams are independent. Summed configuration bps are not portfolio returns; no shared capital, position concurrency, margin or exposure model is implied.