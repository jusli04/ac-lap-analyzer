"""
history.py
Session history manager for AC Lap Analyzer.

Saves a summary of every recording session to session_history.json.
Provides helpers for loading and querying history for the dashboard.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

from analyzer import LapAnalysis, format_lap_time
from ml_coach import MLCoachState, compute_corner_stats


HISTORY_PATH = Path("session_history.json")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CornerSessionSummary:
    corner_number: int
    avg_apex_speed: float
    best_apex_speed: float
    importance_score: float


@dataclass
class SessionRecord:
    session_id:       str        # ISO timestamp used as unique ID
    date:             str        # Human-readable: "2024-03-15 14:32"
    track:            str
    car:              str
    laps_driven:      int
    best_lap_ms:      int
    avg_lap_ms:       int
    best_lap_number:  int
    total_issues:     int        # across all laps
    avg_apex_speed:   float
    top_insight:      str        # single most important ML finding
    corner_summaries: List[CornerSessionSummary] = field(default_factory=list)
    # Raw per-corner ML data for cross-session aggregation
    # corner_number (str key) -> list of {apex_speed, entry_speed, exit_speed, brake_dist, max_brake, lap_time_ms}
    ml_corner_history: Dict[str, List[dict]] = field(default_factory=dict)

    def best_lap_str(self) -> str:
        return format_lap_time(self.best_lap_ms)

    def avg_lap_str(self) -> str:
        return format_lap_time(self.avg_lap_ms)


# ── Build a session record ────────────────────────────────────────────────────

def build_session_record(
    analyses: List[LapAnalysis],
    ml_state: MLCoachState,
    track: str = "Unknown Track",
    car: str = "Unknown Car",
) -> SessionRecord:
    """
    Build a SessionRecord from a completed recording session.
    """
    if not analyses:
        raise ValueError("No laps to summarize")

    now = datetime.now()
    session_id = now.isoformat()
    date_str   = now.strftime("%Y-%m-%d %H:%M")

    best      = min(analyses, key=lambda a: a.lap_time_ms)
    avg_ms    = int(sum(a.lap_time_ms for a in analyses) / len(analyses))
    avg_apex  = sum(a.avg_apex_speed for a in analyses) / len(analyses)
    total_iss = sum(a.total_issues for a in analyses)

    # Top ML insight
    corner_stats = compute_corner_stats(ml_state)
    if corner_stats:
        top = corner_stats[0]
        top_insight = (
            f"Corner {top.corner_number} has the biggest impact on lap time "
            f"({top.importance_score:.0%} importance, "
            f"~{top.lap_time_gain_per_kmh:.2f}s per km/h at apex)."
        )
    else:
        top_insight = "Not enough data for ML insights yet (need 5+ laps)."

    # Corner summaries
    corner_summaries = []
    for cs in corner_stats:
        history = ml_state.corner_history.get(cs.corner_number, [])
        best_apex = max((h["apex_speed"] for h in history), default=0)
        corner_summaries.append(CornerSessionSummary(
            corner_number   = cs.corner_number,
            avg_apex_speed  = cs.avg_apex_speed,
            best_apex_speed = best_apex,
            importance_score= cs.importance_score,
        ))

    # Persist raw ML corner history (convert int keys to str for JSON)
    ml_corner_history = {
        str(k): v for k, v in ml_state.corner_history.items()
    }

    return SessionRecord(
        session_id        = session_id,
        date              = date_str,
        track             = track,
        car               = car,
        laps_driven       = len(analyses),
        best_lap_ms       = best.lap_time_ms,
        avg_lap_ms        = avg_ms,
        best_lap_number   = best.lap_number,
        total_issues      = total_iss,
        avg_apex_speed    = avg_apex,
        top_insight       = top_insight,
        corner_summaries  = corner_summaries,
        ml_corner_history = ml_corner_history,
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def save_session(record: SessionRecord, path: Path = HISTORY_PATH):
    """Append a session record to the history JSON file."""
    history = load_history(path)
    # Replace if same session_id exists (re-run scenario)
    history = [h for h in history if h["session_id"] != record.session_id]
    history.append(asdict(record))
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_history(path: Path = HISTORY_PATH) -> List[dict]:
    """Load all session records from the history JSON file."""
    if not path.exists():
        return []
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def load_sessions(path: Path = HISTORY_PATH) -> List[SessionRecord]:
    """Load history as SessionRecord objects, sorted oldest first."""
    raw = load_history(path)
    sessions = []
    for r in raw:
        try:
            r["corner_summaries"] = [
                CornerSessionSummary(**cs) for cs in r.get("corner_summaries", [])
            ]
            # ml_corner_history may be missing in older saved sessions — default to {}
            r.setdefault("ml_corner_history", {})
            sessions.append(SessionRecord(**r))
        except Exception:
            continue
    sessions.sort(key=lambda s: s.session_id)
    return sessions


# ── Cross-session ML state reconstruction ────────────────────────────────────

def load_ml_state_for_combo(
    track: str,
    car: str,
    path: Path = HISTORY_PATH,
) -> MLCoachState:
    """
    Build an MLCoachState by pooling raw corner data from ALL saved sessions
    that match the given car + track combination (case-insensitive).

    This is the key function that makes ML insights work across sessions:
    instead of needing 5+ laps in one go, any 5 laps on the same car+track
    over any number of sessions will unlock the ML coach.
    """
    sessions = load_sessions(path)
    state = MLCoachState()

    track_norm = track.strip().lower()
    car_norm   = car.strip().lower()

    matched = 0
    for s in sessions:
        if s.track.strip().lower() != track_norm:
            continue
        if s.car.strip().lower() != car_norm:
            continue

        # Merge this session's corner history into the combined state
        for corner_key, entries in s.ml_corner_history.items():
            corner_num = int(corner_key)
            if corner_num not in state.corner_history:
                state.corner_history[corner_num] = []
            state.corner_history[corner_num].extend(entries)
        matched += 1

    if matched > 0:
        total_laps = max((len(h) for h in state.corner_history.values()), default=0)
        print(
            f"[ML] Loaded cross-session history: {matched} sessions matched "
            f"'{car}' at '{track}' — {total_laps} laps total in ML state."
        )

    return state


def total_laps_for_combo(
    track: str,
    car: str,
    path: Path = HISTORY_PATH,
) -> int:
    """
    Return the total number of ML-recorded laps across all sessions
    for a given car+track combination. Used to show the user how close
    they are to unlocking ML insights.
    """
    state = load_ml_state_for_combo(track, car, path)
    return max((len(h) for h in state.corner_history.values()), default=0)


# ── Analytics helpers ─────────────────────────────────────────────────────────

def progression_data(sessions: List[SessionRecord]) -> dict:
    """
    Returns data for the best lap time progression chart.
    {dates, best_laps_s, avg_laps_s, tracks}
    """
    return {
        "dates":       [s.date for s in sessions],
        "best_laps_s": [s.best_lap_ms / 1000 for s in sessions],
        "avg_laps_s":  [s.avg_lap_ms  / 1000 for s in sessions],
        "tracks":      [s.track for s in sessions],
        "laps_driven": [s.laps_driven for s in sessions],
    }


def corner_history_heatmap(sessions: List[SessionRecord]) -> dict:
    """
    Build data for a corner consistency heatmap:
    rows = sessions, cols = corner numbers, values = avg apex speed.
    Returns {session_labels, corner_labels, matrix}
    """
    # Collect all corner numbers seen across all sessions
    all_corners = sorted(set(
        cs.corner_number
        for s in sessions
        for cs in s.corner_summaries
    ))

    matrix = []
    session_labels = []
    for s in sessions:
        corner_map = {cs.corner_number: cs.avg_apex_speed for cs in s.corner_summaries}
        row = [corner_map.get(c, None) for c in all_corners]
        matrix.append(row)
        session_labels.append(s.date)

    return {
        "session_labels": session_labels,
        "corner_labels":  [f"C{c}" for c in all_corners],
        "matrix":         matrix,
    }


def personal_bests(sessions: List[SessionRecord]) -> dict:
    """Overall personal bests across all sessions."""
    if not sessions:
        return {}
    best_session = min(sessions, key=lambda s: s.best_lap_ms)
    return {
        "best_lap_ms":      best_session.best_lap_ms,
        "best_lap_session": best_session.date,
        "best_lap_track":   best_session.track,
        "total_laps":       sum(s.laps_driven for s in sessions),
        "total_sessions":   len(sessions),
    }