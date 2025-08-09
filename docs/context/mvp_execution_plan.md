# WhisperChart – MVP Execution Plan

Last updated: 2025-08-09

## Scope (MVP 0.1)
- Add Prediction Line (toggle) rendered over candlesticks using recent data + placeholder forecast.
- Add Confidence Ribbon (toggle, default off) as shaded band around Prediction Line using placeholder bounds.
- Maintain current live updates and lightweight performance.

## User Stories & Acceptance
- Toggle Prediction Line: As a user, I can enable/disable a forecast line. Visible over current chart without lag (<200ms on toggle).
- Toggle Confidence Ribbon: As a user, I can enable/disable an uncertainty band. Defaults off to reduce clutter.
- Future Projection: The forecast extends to the end of the current trading session (placeholder OK in MVP).

## Architecture Outline
- UI (Streamlit): add toggles and render additional `series` for prediction and ribbon via `streamlit-lightweight-charts`.
- Forecast Provider: lightweight placeholder function now; pluggable model provider later.
- Data Flow: reuse existing bars; generate prediction series from last N bars; compute ribbon bounds.

## Model I/O Contract (for future real model)
- Input: window of OHLCV bars (DataFrame), horizon (minutes/steps), optional features (volatility, momentum, news sentiment), and clock time.
- Output:
  - `y_hat`: list[float] future close prices per step
  - `lower`: list[float] lower bound per step
  - `upper`: list[float] upper bound per step
  - `weight`: list[float] certainty weights in [0,1] (optional)

## Implementation Plan (Incremental)
1) UI toggles (low risk)
   - Add sidebar controls: `show_prediction: bool`, `show_ribbon: bool` (default False), `horizon_minutes: int`.
   - Wire toggles into chart rendering path.

2) Placeholder forecast provider
   - New module `app/predictor.py`:
     - `predict_placeholder(bars: pd.DataFrame, horizon_min: int) -> dict` returning y_hat/lower/upper.
     - Use last-K linear regression or EMA extrapolation with simple CI (e.g., recent volatility).

3) Chart rendering
   - Build additional `series`:
     - Line series for `y_hat` with future timestamps (Unix seconds).
     - Area/band representation for bounds (approximation with two area series is acceptable in MVP).
   - Extend chart’s timeScale to include future points.

4) Performance & UX polish
   - Cache placeholder forecast for TTL (e.g., 15–30s) keyed by symbol/timeframe/horizon.
   - Ensure toggling is snappy; avoid recomputation if cache valid.

## File Changes (proposed)
- `app/app.py`: add toggles + render prediction/ribbon series.
- `app/predictor.py`: placeholder forecast logic.
- `AGENTS.md`: note handoff boundaries for model integration.
- (Optional) `docs/context/adr/` to track decisions as ADRs.

## Risks & Mitigations
- Visual clutter: default ribbon off; use subtle colors; keep line thin.
- Time alignment: ensure future timestamps start after last bar and extend to session close.
- Mobile perf: keep series sizes small; downsample for long horizons.

## Definition of Done
- Toggles present and functional.
- Prediction Line + Ribbon render without errors for SPY on 1m/5m data.
- No regressions in existing chart.
- Lint/format pass; docs updated.

## Next (Post‑MVP)
- Real model integration (LSTM/Transformer)
- Scenario sliders (volatility, drift) to perturb predictions
- Confidence ribbon sourced from predictive uncertainty
- Simple onboarding tooltip flow

