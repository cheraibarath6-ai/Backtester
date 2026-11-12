# Model 3 Crypto Futures Backtest

Research-only copy of the Model 3 multi-candle limit-retracement backtest.

- BTCUSDT and ETHUSDT USD-M perpetual futures
- 5m, 15m, 30m and 1h execution bars
- Exact-price retracement limit entries; order may fill on any subsequent candle
- Pending-order expiry: 3, 6 and 10 bars (additional test dimension)
- Retracement grid: 75%, 78.6%, 80%, 82%, 85%
- RR: 2R, 3R, 4R, 5R and 6R
- Chronological samples: 2021–2023 training, 2024, 2025 and 2026 YTD forward periods
- Fees, slippage and official Binance Vision funding-rate archives included
- Ambiguous stop/target same-bar outcomes are resolved conservatively as stop-first

No exchange credentials or private trading code belong in this repository.

Run with GitHub Actions on the public standard `ubuntu-latest` runner. The core matrix contains 16,200 unique configurations per asset (32,400 BTC/ETH asset-level configurations). Including three pending-expiry settings yields 97,200 asset/expiry evaluations before the four chronological sample slices. Results are uploaded as a workflow artifact.
