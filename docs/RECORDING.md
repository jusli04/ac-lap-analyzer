# Recording System

The recording system reads live telemetry from Assetto Corsa via Windows shared memory, samples it at 20 Hz, detects lap boundaries, and serializes completed laps for analysis.

---

## How It Works

Assetto Corsa exposes three shared memory regions on Windows:

| Region | Contents |
|--------|----------|
| `Local\acpmf_physics` | Speed, throttle, brake, steering angle, G-forces, RPM, gear |
| `Local\acpmf_graphics` | Track position, lap count, lap time, session state |
| `Local\acpmf_static` | Car model, track name, session type (read once at connect) |

`ac_telemetry.py` maps these using Python's `mmap` + `ctypes` and polls them at a configurable Hz rate (default 20 Hz = one sample every 50ms).

---

## Lap Boundary Detection

Lap boundaries are detected by watching `gfx.completedLaps` (the integer lap counter AC increments at the finish line). When it increases:

1. The current in-progress sample buffer is finalized as a completed lap
2. `gfx.iLastTime` is read **immediately** at the boundary — this is AC's own lap timer, accurate to the millisecond, and more reliable than computing elapsed time from samples
3. A new buffer is started for the next lap

**Pit exit detection:** Recording ignores samples until the car exceeds 80 km/h, preventing garbage data from pit lane crawls at session start from polluting lap 1.

**Partial lap discard:** Any buffer that never triggered a lap boundary (e.g. the incomplete final lap when you stop recording mid-lap) is discarded.

---

## Sample Structure

Each `TelemetrySample` captures:

```python
@dataclass
class TelemetrySample:
    timestamp_ms: int        # ms since recording started
    lap_time_ms: int         # AC's current lap timer (iCurrentTime)
    track_position: float    # normalized 0.0–1.0 around the track
    speed_kmh: float
    throttle: float          # 0.0–1.0
    brake: float             # 0.0–1.0
    steer_angle: float       # degrees
    gear: int
    rpms: int
    g_lat: float             # lateral G (cornering)
    g_lon: float             # longitudinal G (accel/brake)
    x: float                 # GPS X coordinate
    z: float                 # GPS Z coordinate (depth)
```

At 20 Hz, a 90-second lap produces ~1,800 samples. Each sample is ~100 bytes, so a full session of 10 laps is about 1.8 MB in memory before pickling.

---

## Dashboard Recording

The dashboard's **Live Record** page runs the recorder in a background thread so the UI stays responsive:

```python
stop_evt = threading.Event()
recorder = LapRecorder(sample_rate_hz=20)

def _run():
    laps = recorder.record(duration_seconds=duration, stop_event=stop_evt)
    st.session_state.recorder_laps = laps or []
    st.session_state.recording = False

threading.Thread(target=_run, daemon=True).start()
```

Clicking **Stop Recording** sets `stop_evt`, which the recorder's inner loop checks on every sample tick — stopping cleanly without killing the Streamlit process.

---

## CLI Recording

`record.py` provides a standalone CLI recorder with post-session prompts:

```bash
python record.py                    # 5-minute max session
python record.py --duration 600     # 10-minute max session
python record.py --hz 30            # 30 Hz sample rate
```

After recording, it prints a quick analysis summary and prompts you to save the session to history with a track and car name — which feeds the ML coach's cross-session model.

---

## Persistence

Completed laps are pickled to `saved_laps.pkl` immediately after recording. This file is overwritten each session — it holds only the most recent session's raw laps.

Long-term history (across sessions) is stored in `session_history.json` via the **Save Session** flow on the History page. That file accumulates `SessionRecord` objects which include per-corner ML feature vectors for the cross-session model.

---

## Ghost Lap Filtering

Occasionally AC fires a spurious lap boundary event (e.g. at session start before the car has moved). These produce laps with lap times of a few milliseconds. The dashboard filters these out:

```python
MIN_LAP_MS = 10_000  # ignore laps under 10 seconds
clean_laps = [lap for lap in laps if lap[-1].lap_time_ms > MIN_LAP_MS]
```
