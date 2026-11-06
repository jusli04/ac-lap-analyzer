"""
ml_coach.py
Machine learning coaching layer for AC Lap Analyzer.

Needs at least 5 laps to start working, gets much better at 20+.
Provides:
  - Lap time prediction from corner inputs
  - Corner importance (which corners explain the most lap time variance)
  - Anomaly detection (flag outlier corners vs your average)
  - Sector regression: "gaining X seconds at corner N = Y seconds on lap time"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from analyzer import LapAnalysis, Corner, analyze_lap


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CornerStats:
    corner_number: int
    avg_apex_speed: float
    std_apex_speed: float
    avg_entry_speed: float
    avg_exit_speed: float
    avg_brake_distance: float  # in samples
    importance_score: float    # 0-1, how much this corner affects lap time
    lap_time_gain_per_kmh: float  # seconds gained per 1 km/h faster at apex


@dataclass
class LapMLInsights:
    predicted_lap_time_ms: int
    actual_lap_time_ms: int
    prediction_error_ms: int          # actual - predicted (negative = faster than predicted)
    corner_importances: List[Tuple[int, float]]  # (corner_number, importance 0-1)
    anomalous_corners: List[Tuple[int, str]]     # (corner_number, description)
    gains_available: List[Tuple[int, float]]     # (corner_number, seconds available)
    summary: str


@dataclass
class MLCoachState:
    """Persistent state — grows as you record more laps."""
    corner_history: Dict[int, List[dict]] = field(default_factory=dict)
    # corner_number -> list of {apex_speed, entry_speed, exit_speed, brake_dist, lap_time_ms}
    model_trained: bool = False
    min_laps_for_model: int = 5


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_corner_features(analysis: LapAnalysis, samples) -> dict:
    """
    Extract per-corner features from a lap analysis for ML training.
    Returns dict of corner_number -> feature dict.
    """
    features = {}
    for i, corner in enumerate(analysis.corners):
        brake_dist = 0
        if corner.brake_start_idx is not None:
            brake_dist = corner.entry_idx - corner.brake_start_idx

        features[i + 1] = {
            "apex_speed":    corner.min_speed_kmh,
            "entry_speed":   corner.entry_speed_kmh,
            "exit_speed":    corner.exit_speed_kmh,
            "brake_dist":    brake_dist,
            "max_brake":     corner.max_brake,
            "lap_time_ms":   analysis.lap_time_ms,
        }
    return features


def update_history(state: MLCoachState, analysis: LapAnalysis, samples) -> MLCoachState:
    """Add a new lap's corner data to the historical state."""
    features = extract_corner_features(analysis, samples)
    for corner_num, feat in features.items():
        if corner_num not in state.corner_history:
            state.corner_history[corner_num] = []
        state.corner_history[corner_num].append(feat)
    return state


# ── Corner statistics ─────────────────────────────────────────────────────────

def compute_corner_stats(state: MLCoachState) -> List[CornerStats]:
    """
    Compute statistics for each corner from historical data.
    Importance score = correlation of apex speed with lap time.
    """
    stats = []

    for corner_num, history in state.corner_history.items():
        if len(history) < 3:
            continue

        apex_speeds  = np.array([h["apex_speed"]  for h in history])
        entry_speeds = np.array([h["entry_speed"] for h in history])
        exit_speeds  = np.array([h["exit_speed"]  for h in history])
        brake_dists  = np.array([h["brake_dist"]  for h in history])
        lap_times    = np.array([h["lap_time_ms"] for h in history])

        # Correlation between apex speed and lap time
        # Higher apex speed → lower lap time (negative correlation = more important)
        if apex_speeds.std() > 0 and lap_times.std() > 0:
            corr = np.corrcoef(apex_speeds, lap_times)[0, 1]
            importance = abs(corr)  # 0-1
        else:
            importance = 0.0
            corr = 0.0

        # Linear regression: how many ms do we gain per 1 km/h at apex?
        # lap_time = a * apex_speed + b  →  slope = ms per km/h
        if apex_speeds.std() > 0:
            slope = np.polyfit(apex_speeds, lap_times, 1)[0]
            gain_per_kmh = -slope / 1000  # convert ms to seconds, flip sign
        else:
            gain_per_kmh = 0.0

        stats.append(CornerStats(
            corner_number          = corner_num,
            avg_apex_speed         = float(apex_speeds.mean()),
            std_apex_speed         = float(apex_speeds.std()),
            avg_entry_speed        = float(entry_speeds.mean()),
            avg_exit_speed         = float(exit_speeds.mean()),
            avg_brake_distance     = float(brake_dists.mean()),
            importance_score       = float(importance),
            lap_time_gain_per_kmh  = float(gain_per_kmh),
        ))

    stats.sort(key=lambda s: s.importance_score, reverse=True)
    return stats


# ── Lap time prediction ───────────────────────────────────────────────────────

def predict_lap_time(state: MLCoachState, analysis: LapAnalysis) -> Optional[int]:
    """
    Simple linear model: predicted_lap_time = sum(corner_contribution) + baseline.
    Uses per-corner regression trained on historical data.
    Returns predicted lap time in ms, or None if not enough data.
    """
    if not state.corner_history:
        return None

    total_predicted_ms = 0
    corners_used = 0
    baseline_per_corner = 0

    for i, corner in enumerate(analysis.corners):
        corner_num = i + 1
        history = state.corner_history.get(corner_num, [])

        if len(history) < 3:
            continue

        apex_speeds = np.array([h["apex_speed"]  for h in history])
        lap_times   = np.array([h["lap_time_ms"] for h in history])

        if apex_speeds.std() > 0:
            slope, intercept = np.polyfit(apex_speeds, lap_times, 1)
            predicted = slope * corner.min_speed_kmh + intercept
            total_predicted_ms += predicted
            corners_used += 1

    if corners_used == 0:
        return None

    # Average across corners (each predicts full lap time, we average)
    return int(total_predicted_ms / corners_used)


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_anomalies(state: MLCoachState,
                     analysis: LapAnalysis,
                     z_threshold: float = 2.0) -> List[Tuple[int, str]]:
    """
    Flag corners where this lap's performance is > z_threshold standard deviations
    from the historical mean. Returns list of (corner_number, description).
    """
    anomalies = []

    for i, corner in enumerate(analysis.corners):
        corner_num = i + 1
        history = state.corner_history.get(corner_num, [])

        if len(history) < 4:
            continue

        apex_speeds = np.array([h["apex_speed"] for h in history])
        mean = apex_speeds.mean()
        std  = apex_speeds.std()

        if std < 0.5:  # too little variance to say anything
            continue

        z = (corner.min_speed_kmh - mean) / std

        if z < -z_threshold:
            anomalies.append((
                corner_num,
                f"Apex speed {corner.min_speed_kmh:.1f} km/h — "
                f"{abs(z):.1f}σ below your average ({mean:.1f} km/h). "
                f"Something went wrong here."
            ))
        elif z > z_threshold:
            anomalies.append((
                corner_num,
                f"Apex speed {corner.min_speed_kmh:.1f} km/h — "
                f"{z:.1f}σ above your average ({mean:.1f} km/h). "
                f"Personal best at this corner!"
            ))

    return anomalies


# ── Gains available ───────────────────────────────────────────────────────────

def compute_gains_available(state: MLCoachState,
                            analysis: LapAnalysis,
                            top_n: int = 3) -> List[Tuple[int, float]]:
    """
    For each corner, compute how many seconds are available if you hit your
    personal best apex speed at that corner. Returns top N corners by gain.
    """
    gains = []

    for i, corner in enumerate(analysis.corners):
        corner_num = i + 1
        history = state.corner_history.get(corner_num, [])

        if len(history) < 3:
            continue

        apex_speeds = np.array([h["apex_speed"] for h in history])
        lap_times   = np.array([h["lap_time_ms"] for h in history])
        best_apex   = apex_speeds.max()

        if apex_speeds.std() < 0.5:
            continue

        delta_speed = best_apex - corner.min_speed_kmh
        if delta_speed <= 0:
            continue

        # Use regression slope to convert speed delta → time delta
        if apex_speeds.std() > 0:
            slope = np.polyfit(apex_speeds, lap_times, 1)[0]
            gain_ms = -slope * delta_speed  # negative slope means faster = lower time
            gain_s  = gain_ms / 1000
            if gain_s > 0.05:  # only report gains > 50ms
                gains.append((corner_num, gain_s))

    gains.sort(key=lambda g: g[1], reverse=True)
    return gains[:top_n]


# ── Full ML analysis ──────────────────────────────────────────────────────────

def analyze_lap_ml(state: MLCoachState, analysis: LapAnalysis, samples) -> Optional[LapMLInsights]:
    """
    Run the full ML coaching pipeline on a completed lap.
    Returns None if not enough historical data yet.
    """
    total_laps = max((len(h) for h in state.corner_history.values()), default=0)
    if total_laps < state.min_laps_for_model:
        return None

    predicted = predict_lap_time(state, analysis)
    if predicted is None:
        return None

    corner_stats  = compute_corner_stats(state)
    anomalies     = detect_anomalies(state, analysis)
    gains         = compute_gains_available(state, analysis)

    # Corner importances sorted
    importances = [(s.corner_number, s.importance_score) for s in corner_stats]

    # Build summary
    top_corner = corner_stats[0] if corner_stats else None
    total_gain = sum(g for _, g in gains)

    error_ms = analysis.lap_time_ms - predicted
    if abs(error_ms) < 500:
        prediction_note = "consistent with your pattern"
    elif error_ms > 0:
        prediction_note = f"{abs(error_ms/1000):.2f}s slower than predicted"
    else:
        prediction_note = f"{abs(error_ms/1000):.2f}s faster than predicted"

    lines = [f"Lap {analysis.lap_number} is {prediction_note}."]

    if top_corner:
        lines.append(
            f"Corner {top_corner.corner_number} has the biggest impact on your lap time "
            f"(importance {top_corner.importance_score:.0%}). "
            f"Every 1 km/h faster at the apex here is worth ~{top_corner.lap_time_gain_per_kmh:.2f}s."
        )

    if gains:
        lines.append(
            f"Up to {total_gain:.2f}s available if you match your personal bests — "
            f"biggest opportunity at corner {gains[0][0]} ({gains[0][1]:.2f}s)."
        )

    if anomalies:
        bad = [a for a in anomalies if "went wrong" in a[1]]
        good = [a for a in anomalies if "best" in a[1]]
        if bad:
            lines.append(f"⚠ Corner {bad[0][0]} was unusually slow this lap.")
        if good:
            lines.append(f"✅ Corner {good[0][0]} was a personal best this lap!")

    return LapMLInsights(
        predicted_lap_time_ms  = predicted,
        actual_lap_time_ms     = analysis.lap_time_ms,
        prediction_error_ms    = error_ms,
        corner_importances     = importances,
        anomalous_corners      = anomalies,
        gains_available        = gains,
        summary                = " ".join(lines),
    )


def format_lap_time(ms: int) -> str:
    minutes = ms // 60000
    seconds = (ms % 60000) / 1000
    return f"{minutes}:{seconds:06.3f}"
