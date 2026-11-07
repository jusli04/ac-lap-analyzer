# ML Coach

The ML coach analyzes your driving patterns across multiple sessions and delivers personalized coaching insights: which corners cost you the most time, how close you are to your personal best in each corner, and when a corner goes significantly worse than your average.

---

## Overview

The model is intentionally lightweight — NumPy linear regression and z-score anomaly detection — because the dataset is small (tens to hundreds of laps) and interpretability matters more than accuracy. Every output the coach gives can be traced back to a specific number in your data.

---

## Corner Feature Extraction

After each lap is recorded, the analyzer segments it into corners (see `analyzer.py`) and extracts a feature vector for each corner:

| Feature | Description |
|---------|-------------|
| `apex_speed_kmh` | Speed at the corner's apex (slowest point mid-corner) |
| `entry_speed_kmh` | Speed at corner entry |
| `exit_speed_kmh` | Speed at corner exit |
| `min_speed_kmh` | Minimum speed through the entire corner zone |
| `brake_pressure` | Peak brake input during the braking zone |
| `throttle_exit` | Throttle application at corner exit |
| `braking_distance` | Number of samples from brake start to corner entry |
| `g_lat_max` | Peak lateral G-force through the corner |

These are stored in `MLCoachState.corner_history` — a dict mapping `corner_id → List[feature_vector]` — and serialized to `session_history.json` for cross-session persistence.

---

## Linear Regression Model

Once 5+ laps are available for a car/track combination, the coach fits a linear regression model:

```
lap_time_ms = β₀ + β₁·apex_C1 + β₂·apex_C2 + ... + βₙ·apex_Cₙ + ε
```

**Input:** apex speed (km/h) for each corner across all recorded laps  
**Output:** predicted lap time (ms)

This is fit using NumPy's least-squares solver (`np.linalg.lstsq`). The model is re-fit from scratch on every prediction using all available history — no incremental updates.

### Corner Importance

The regression coefficients (β values) tell us how much each corner's apex speed affects lap time. A larger negative coefficient means going faster through that corner saves more time per km/h.

Importance scores are normalized to [0, 1]:

```python
raw_importance = abs(coefficient) * std(apex_speeds_for_corner)
importance = raw_importance / sum(all_importances)
```

Multiplying by standard deviation weights corners where you have more variance (more room to improve) higher than corners where you're already consistent.

### Personal Best Targets

For each corner, the model computes:

```
gain_available = (personal_best_apex - current_apex) × |coefficient|
```

This estimates how many seconds you'd gain on your lap time if you matched your own personal best apex speed in every corner simultaneously.

### Lap Time Prediction

The fitted model predicts what your lap time *should* have been given your corner apex speeds. The delta between predicted and actual highlights whether a lap was better or worse than your cornering inputs suggest — useful for detecting setup or tyre issues.

---

## Anomaly Detection

For each corner on each lap, the coach computes a z-score against your historical distribution for that corner:

```
z = (apex_speed - mean_apex) / std_apex
```

| z-score | Interpretation |
|---------|---------------|
| z < −2.0 | 🔴 Significantly worse than average — flagged as anomaly |
| −2.0 ≤ z ≤ 1.5 | Normal range |
| z > 1.5 | ✅ Better than average — flagged as positive |

The threshold is intentionally asymmetric: anomalies (bad corners) require a larger deviation to fire than positives (good corners), to avoid alert fatigue.

---

## Cross-Session Persistence

The ML coach's value compounds across sessions. Feature vectors are serialized alongside session metadata in `session_history.json`:

```json
{
  "sessions": [...],
  "ml_states": {
    "brands_hatch_indy__ferrari_488_gt3": {
      "corner_history": {
        "0": [[apex, entry, exit, ...], [apex, entry, exit, ...], ...],
        "1": [...],
        ...
      }
    }
  }
}
```

The key is `{track}__{car}` (normalized to lowercase, spaces replaced with underscores). Loading the History page and selecting a track/car combination pools all saved feature vectors for that combo, regardless of how many sessions they span.

---

## Unlock Threshold

The ML coach requires **5 or more laps** on the same car/track combination before producing insights. Below that threshold, the regression is unreliable (more coefficients than meaningful observations).

The threshold is configurable:

```python
ml_state.min_laps_for_model = 5  # default
```

---

## Limitations

- **Small dataset:** With 5–20 laps, the regression is noisy. Corner importance rankings stabilize around 15–20 laps.
- **Single feature per corner:** The model uses only apex speed as the predictor. A more sophisticated model would include entry speed, braking point, and throttle application.
- **No track sector model:** The model treats all corners as independent. In reality, a slow corner 2 affects corner 3's entry speed — this interaction is not captured.
- **Linear assumption:** The relationship between apex speed and lap time is approximately linear within a narrow speed range, but breaks down across very different driving styles or setups.

These are deliberate simplifications for a personal project. The same architecture (feature extraction → regression → anomaly detection) scales directly to production telemetry systems with richer feature sets and larger datasets.
