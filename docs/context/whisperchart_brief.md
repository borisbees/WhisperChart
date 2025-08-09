# WhisperChart – Working Product Brief

Last updated: 2025-08-09

## Core Idea
An AI-powered, beginner‑friendly trading tool that focuses less on perfect prediction and more on building trading intuition via guided, interactive visualizations.

## Main Product Goals
- Clarity AND accuracy: avoid overwhelming beginners with complex analytics, however make sure that were still accurate and reliable as users gain skill. 
- Learning-first: scenario exploration and guidance to understand patterns in real time live market data.
- Continuous feedback: prediction behaves like a living signal updating in real time.
- Confidence visualization: clearly show uncertainty and accuracy.
- User control: toggle advanced visuals on/off.

## Target Audience
- Primary: Regular retail traders wanting a quick, reliable, and accurate forecasting tool that shows predicted intraday and interday price activity in real time, to guide them in making high POP trades and making a ton of money.
- Secondary: intermediates validating intuition and beginner traders building market reading skills.
- Tertiary: educators/content creators embedding charts into lessons.

## Key UI Features
- Prediction Line: continuously updated, central guidance path.
- Confidence Ribbon (toggle, off by default): shaded certainty around predictions; extends to end of trading day.
- Scenario Exploration (roadmap): adjustable variables (e.g., volatility, momentum) to compare outcomes.
- Beginner‑first UX: minimal clutter, simple terminology, visual cues over jargon.

## Design Philosophy
- No binary buy/sell calls; show an evolving forecast instead.
- Transparency: uncertainty is visible, not hidden.
- Gamified learning: what‑if exploration + compare predictions vs. outcomes.
- Progressive disclosure: start simple; reveal complexity as comfort grows.

## Technical Considerations
- Model output needs:
  - Continuous short‑horizon price path (sequence).
  - Confidence intervals for each horizon step (lower/upper bounds).
  - Certainty weighting for heatmap/ribbon strength.
  - some type of continous learning - RL, etc
- Chart UI needs:
  - Real‑time updates, light and fast (mobile‑friendly if possible).
  - Confidence shading and future projection to session close.
  - Toggles for ribbons and complexity.

## Non‑Goals (for MVP)
- Full broker integration and order routing.
- High‑latency, heavyweight models in the request path.
- Exhaustive technical indicator dashboards.

## Success Criteria (early)
- App loads fast (<2–3s) and stays responsive.
- Users can turn on/off Prediction Line and Confidence Ribbon.
- Ribbon clearly communicates uncertainty without clutter.
- Users report better understanding of market movement after short use.

---

Sources: Consolidated from prior WhisperChart discussions and current repo direction.
