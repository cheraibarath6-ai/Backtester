# Model 3 Full-Universe Asset-Specific Search

This workflow evaluates the complete pre-defined Model 3 parameter grid independently on ten Binance USD-M perpetual assets. It does not transfer BTC/ETH-ranked candidates to other assets and does not use multi-model consensus.

## Complete candidate universe

Base parameter dimensions:
- Timeframes: 5m, 15m, 30m, 1h
- ATR periods: 14, 20, 25
- Impulse thresholds: 1.8, 2.0, 2.2, 2.4, 2.5, 2.7 ATR
- ADX bands: 18–30, 15–25, 20–35
- RSI modes: exact, loose, none
- Retracement levels: 75%, 78.6%, 80%, 82%, 85%
- RR: 2R, 3R, 4R, 5R, 6R
- Pending expiry: 3, 6, 10 bars

This is 3,240 base parameter sets, 48,600 complete configurations per asset after RR and expiry, and 486,000 asset-specific configuration tests across ten assets. Each configuration is replayed over 2021–2023 development and 2024, 2025 and 2026 YTD evaluation periods.

## Asset-specific selection

Each asset is ranked independently using 2021–2023 only. The preferred pool has at least 100 trades, positive net bps after costs/funding, and maximum drawdown no worse than 35%. If fewer than two candidates meet all preferred conditions, the selection falls back progressively to the best available candidates and marks that fallback explicitly. There is no universal PF cutoff that forces an asset to have no candidate, and no out-of-sample result is used to choose the candidates.

The selection score balances log(PF), trade count, net development return and drawdown. Every selected configuration remains an independent model and alert source. No voting, consensus, ensemble, cross-asset confirmation or mandatory agreement is applied.

## Execution and data

- Official Binance USD-M 1-minute klines, resampled to the tested timeframe.
- Official Binance Vision fundingRate archives only.
- Data preflight checks missing minutes, duplicate timestamps, invalid OHLC rows and suspicious funding settlement gaps. It fails rather than silently imputing missing candles or assuming zero funding.
- 5 bps fee and 2 bps slippage per side.
- Exact-price multi-candle retracement limit entry, valid for the configured pending bars.
- Stop-first handling when stop and target are both touched in one candle.
- Existing timeframe-specific maximum holding periods.

## Reporting

The final report contains:
- All candidate metrics for all ten assets and all four periods.
- Development ranking for every asset's complete candidate universe.
- Exactly two selected configurations per asset, with full parameters.
- Individual OOS PF, trade count, net bps and drawdown for each selected configuration.
- Combined trade-level PF for each selected pair by year and across all OOS years.

Combined PF is the pooled trade-level PF of the two independent configurations, not a portfolio return. It does not model shared capital, overlapping positions, leverage or execution conflicts. The selected configurations require forward paper validation before any live use.
