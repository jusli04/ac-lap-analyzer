"""
analyzer.py
Corner detection and coaching feedback engine.
Works on a list of TelemetrySample objects from one lap.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from ac_telemetry import TelemetrySample


# ── Corner detection ──────────────────────────────────────────────────────────

@dataclass
class Corner:
    entry_idx:      int
    apex_idx:       int
    exit_idx:       int
    track_pos_entry: float
    track_pos_apex:  float
    track_pos_exit:  float
    min_speed_kmh:   float
    entry_speed_kmh: float
    exit_speed_kmh:  float
    brake_start_idx: Optional[int]
    throttle_app_idx: Optional[int]  # first throttle after apex
    max_steer:       float
    max_brake:       float


def detect_corners(samples: List[TelemetrySample],
                   steer_threshold: float = 0.15,
                   min_corner_samples: int = 20,
                   merge_gap_samples: int = 15) -> List[Corner]:
    """
    Detects corners by finding regions where steering angle exceeds threshold,
    then merges segments that are close together (same physical corner broken
    up by a brief steering correction mid-corner, e.g. Druids at Brands Hatch).

    Parameters
    ----------
    steer_threshold      Minimum absolute steering angle to enter a corner.
                         Raised from 0.08 → 0.15 to ignore minor corrections.
    min_corner_samples   Minimum length of a corner segment in samples (at 20 Hz,
                         20 samples = 1 second). Raised from 8 → 20.
    merge_gap_samples    If two corner segments are separated by fewer than this
                         many samples of straight steering, merge them into one.
                         This prevents long corners (Druids) from splitting.
    """
    # ── Pass 1: raw segment detection ────────────────────────────────────────
    raw_segments = []   # list of (start_idx, end_idx)
    in_corner = False
    corner_start = 0

    for i, s in enumerate(samples):
        steering = abs(s.steer_angle)

        if not in_corner and steering > steer_threshold:
            in_corner = True
            corner_start = i

        elif in_corner and steering <= steer_threshold:
            in_corner = False
            corner_end = i
            if corner_end - corner_start >= min_corner_samples:
                raw_segments.append((corner_start, corner_end))

    # Handle case where lap ends mid-corner
    if in_corner and len(samples) - corner_start >= min_corner_samples:
        raw_segments.append((corner_start, len(samples) - 1))

    # ── Pass 2: merge nearby segments ────────────────────────────────────────
    if not raw_segments:
        return []

    merged = [raw_segments[0]]
    for start, end in raw_segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < merge_gap_samples:
            # Close enough — extend the previous segment
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # ── Pass 3: build Corner objects ─────────────────────────────────────────
    corners = []
    for corner_start, corner_end in merged:
        seg = samples[corner_start:corner_end]
        if not seg:
            continue

        # Apex = minimum speed point
        apex_offset = min(range(len(seg)), key=lambda j: seg[j].speed_kmh)
        apex_idx = corner_start + apex_offset

        # Brake start — search backwards from corner entry
        brake_start = None
        for j in range(corner_start, max(0, corner_start - 60), -1):
            if samples[j].brake > 0.05:
                brake_start = j
                break

        # First throttle after apex
        throttle_app = None
        for j in range(apex_idx, corner_end):
            if samples[j].throttle > 0.1:
                throttle_app = j
                break

        corners.append(Corner(
            entry_idx        = corner_start,
            apex_idx         = apex_idx,
            exit_idx         = corner_end,
            track_pos_entry  = seg[0].track_position,
            track_pos_apex   = seg[apex_offset].track_position,
            track_pos_exit   = seg[-1].track_position,
            min_speed_kmh    = min(s.speed_kmh for s in seg),
            entry_speed_kmh  = seg[0].speed_kmh,
            exit_speed_kmh   = seg[-1].speed_kmh,
            brake_start_idx  = brake_start,
            throttle_app_idx = throttle_app,
            max_steer        = max(abs(s.steer_angle) for s in seg),
            max_brake        = max(s.brake for s in seg),
        ))

    return corners


# ── Coaching feedback ─────────────────────────────────────────────────────────

@dataclass
class CornerFeedback:
    corner_number:  int
    track_position: float
    issues:         List[str] = field(default_factory=list)
    positives:      List[str] = field(default_factory=list)
    overall:        str = ""


def analyze_corner(corner: Corner, corner_number: int,
                   samples: List[TelemetrySample],
                   ref_corner: Optional[Corner] = None) -> CornerFeedback:
    """
    Generates coaching feedback for a single corner.
    Optionally compares to a reference (best) lap corner.
    """
    fb = CornerFeedback(
        corner_number  = corner_number,
        track_position = corner.track_pos_entry,
    )

    # ── Braking analysis ──
    if corner.brake_start_idx is not None:
        brake_sample = samples[corner.brake_start_idx]
        braking_distance_samples = corner.entry_idx - corner.brake_start_idx

        if braking_distance_samples < 5:
            fb.issues.append(
                f"Braking too late — only {braking_distance_samples} samples before corner entry. "
                f"Try braking earlier at track position {brake_sample.track_position:.2f}."
            )
        elif braking_distance_samples > 40:
            fb.issues.append(
                f"Possibly braking too early — {braking_distance_samples} samples before corner. "
                "You may be leaving time on the table. Try committing later."
            )
        else:
            fb.positives.append("Braking point looks reasonable.")

        if corner.max_brake < 0.5 and corner.entry_speed_kmh > 80:
            fb.issues.append(
                "Braking pressure is low for the entry speed. "
                "Apply more brake force early in the braking zone, then trail off."
            )
        elif corner.max_brake > 0.95:
            fb.positives.append("Good hard braking — full brake pressure applied.")

    else:
        if corner.entry_speed_kmh > 60:
            fb.issues.append(
                "No braking detected before this corner. "
                "If this is a high-speed corner that's intentional — otherwise try trail braking."
            )

    # ── Apex speed ──
    if corner.min_speed_kmh < 30:
        fb.issues.append(
            f"Very low apex speed ({corner.min_speed_kmh:.1f} km/h). "
            "You may be over-slowing — trust the car and carry more speed through the apex."
        )

    # ── Throttle application ──
    if corner.throttle_app_idx is not None:
        samples_after_apex = corner.throttle_app_idx - corner.apex_idx
        if samples_after_apex > 15:
            fb.issues.append(
                f"Throttle applied {samples_after_apex} samples after apex — this is late. "
                "Try getting on the power earlier at corner exit to maximize acceleration."
            )
        elif samples_after_apex <= 5:
            fb.positives.append("Good early throttle application after the apex.")
    else:
        fb.issues.append(
            "No throttle detected after apex — make sure you're fully opening the throttle on exit."
        )

    # ── Exit speed ──
    if corner.exit_speed_kmh < corner.entry_speed_kmh * 0.6:
        fb.issues.append(
            f"Exit speed ({corner.exit_speed_kmh:.1f} km/h) is much lower than entry "
            f"({corner.entry_speed_kmh:.1f} km/h). Work on getting the car rotated earlier "
            "so you can apply throttle sooner."
        )

    # ── Reference lap comparison ──
    if ref_corner:
        speed_delta = corner.min_speed_kmh - ref_corner.min_speed_kmh
        if speed_delta < -5:
            fb.issues.append(
                f"Apex speed is {abs(speed_delta):.1f} km/h slower than your best lap. "
                "Focus on carrying more mid-corner speed."
            )
        elif speed_delta > 2:
            fb.positives.append(
                f"Apex speed is {speed_delta:.1f} km/h faster than your best lap — great improvement!"
            )

    # ── Overall summary ──
    if not fb.issues:
        fb.overall = "✅ Clean corner — no major issues detected."
    elif len(fb.issues) == 1:
        fb.overall = "⚠️  One issue to work on."
    else:
        fb.overall = f"⚠️  {len(fb.issues)} issues detected — focus on the most impactful first."

    return fb


# ── Full lap analysis ─────────────────────────────────────────────────────────

@dataclass
class LapAnalysis:
    lap_number:      int
    lap_time_ms:     int
    corners:         List[Corner]
    feedback:        List[CornerFeedback]
    avg_apex_speed:  float
    total_issues:    int


def analyze_lap(samples: List[TelemetrySample],
                lap_number: int,
                ref_samples: Optional[List[TelemetrySample]] = None,
                lap_time_ms: int = 0) -> LapAnalysis:
    """Full lap analysis — detects all corners and generates feedback for each."""

    corners = detect_corners(samples)
    ref_corners = detect_corners(ref_samples) if ref_samples else []

    feedback = []
    for i, corner in enumerate(corners):
        ref_c = ref_corners[i] if i < len(ref_corners) else None
        fb = analyze_corner(corner, i + 1, samples, ref_corner=ref_c)
        feedback.append(fb)

    # Use AC-reported lap time if provided, else fall back to timestamp span
    lap_time = lap_time_ms if lap_time_ms > 0 else (
        (samples[-1].timestamp_ms - samples[0].timestamp_ms) if len(samples) > 1 else 0
    )
    avg_apex = (sum(c.min_speed_kmh for c in corners) / len(corners)) if corners else 0

    return LapAnalysis(
        lap_number     = lap_number,
        lap_time_ms    = lap_time,
        corners        = corners,
        feedback       = feedback,
        avg_apex_speed = avg_apex,
        total_issues   = sum(len(f.issues) for f in feedback),
    )


def format_lap_time(ms: int) -> str:
    minutes = ms // 60000
    seconds = (ms % 60000) / 1000
    return f"{minutes}:{seconds:06.3f}"


def print_analysis(analysis: LapAnalysis):
    """Pretty-print analysis to terminal."""
    print(f"\n{'='*60}")
    print(f"  LAP {analysis.lap_number}  —  {format_lap_time(analysis.lap_time_ms)}")
    print(f"  Corners detected: {len(analysis.corners)}")
    print(f"  Avg apex speed:   {analysis.avg_apex_speed:.1f} km/h")
    print(f"  Total issues:     {analysis.total_issues}")
    print(f"{'='*60}\n")

    for fb in analysis.feedback:
        print(f"  Corner {fb.corner_number}  (track pos {fb.track_position:.2f})")
        print(f"  {fb.overall}")
        for issue in fb.issues:
            print(f"    ⚠  {issue}")
        for pos in fb.positives:
            print(f"    ✓  {pos}")
        print()