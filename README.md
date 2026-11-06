# 🏎️ AC Lap Analyzer

A Python telemetry analysis tool for **Assetto Corsa** that reads live shared memory data, detects corners, and gives you coaching feedback on braking points, apex speed, and throttle application — displayed in a real-time Streamlit dashboard.

![Dashboard preview](dashboard_preview.png)

## Features

- 📡 **Live telemetry** — reads directly from AC shared memory at up to 20Hz
- 🏁 **Automatic lap detection** — saves each lap separately on lap completion
- 🔄 **Corner detection** — identifies corners from steering angle data
- 🎯 **Coaching feedback** per corner:
  - Braking point (too early / too late / missing)
  - Braking pressure analysis
  - Apex speed
  - Throttle application timing
  - Exit speed
- 📊 **Visual dashboard** (Streamlit + Plotly):
  - Speed / throttle / brake trace with corner markers
  - Color-coded track map (speed heatmap)
  - Lap time comparison chart
- 🔁 **Reference lap comparison** — compares each lap against your best

---

## Setup

### Requirements
- Windows PC (AC shared memory is Windows-only)
- Python 3.9+
- Assetto Corsa (Steam)

### Install
```bash
git clone https://github.com/yourusername/ac-lap-analyzer
cd ac-lap-analyzer
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Record laps
Launch Assetto Corsa, start a practice session, then in a separate terminal:

```bash
python record.py
```

Drive your laps. Press `Ctrl+C` when done. Laps are saved to `saved_laps.pkl`.

Options:
```bash
python record.py --duration 600    # record for up to 10 minutes
python record.py --hz 30           # higher sample rate
python record.py --out my_laps.pkl # custom output file
```

### Step 2 — View dashboard
```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501` in your browser.

---

## How it works

### Shared Memory
Assetto Corsa exposes three shared memory blocks on Windows:
- `Local Car Physics` — speed, throttle, brake, steering, G-forces (~333Hz)
- `Local Car Graphics` — lap time, track position, car coordinates, session status
- `Local Car Statics` — car model, track name, sector count (set once per session)

These are read using Python's `mmap` + `ctypes` with no external dependencies.

### Corner Detection
Corners are detected when `|steering_angle| > 0.08 radians` for at least 8 consecutive samples. The apex is the minimum speed point within the corner region.

### Coaching Engine
Each corner is evaluated for:
| Check | Method |
|---|---|
| Brake point | Samples between brake application and corner entry |
| Brake pressure | Max brake value vs entry speed |
| Apex speed | Min speed in corner vs reference lap |
| Throttle timing | Samples between apex and first throttle |
| Exit speed | Exit vs entry speed ratio |

---

## Project Structure

```
ac-lap-analyzer/
├── ac_telemetry.py   # Shared memory reader + lap recorder
├── analyzer.py       # Corner detection + coaching engine
├── dashboard.py      # Streamlit dashboard
├── record.py         # CLI recorder
├── requirements.txt
└── README.md
```

---

## Tech Stack
- **Python** — ctypes, mmap for shared memory
- **Pandas** — telemetry data manipulation
- **Plotly** — interactive charts and track map
- **Streamlit** — dashboard UI

---

## ML Coach

After 5+ laps on the same track, the ML Coach tab unlocks:

- **Corner Importance** — which corners explain the most lap time variance (correlation analysis)
- **Time Available** — seconds you can gain by matching your personal best at each corner
- **Anomaly Detection** — flags corners that are unusually slow or fast vs your historical average
- **Lap Prediction** — predicts your lap time from corner inputs; delta tells you if you over/under-performed your pattern

Gets significantly more accurate after 20+ laps. All local, no API key needed.

## Roadmap
- [ ] Sector time breakdown
- [ ] Compare two laps overlaid on same chart
- [ ] Export lap data to CSV/JSON
- [ ] Auto-detect track name and load reference lap by track
- [ ] Parse AC ghost lap `.ai` files as pro reference benchmark

---

Built by Justin Li — bridging competitive motorsport experience with data engineering.
