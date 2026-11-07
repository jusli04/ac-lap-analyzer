# Gamification System

AC Lap Analyzer includes a full progression system designed to keep you motivated and coming back every day — the same psychological hooks that make games like Snapchat and Duolingo sticky, applied to sim racing improvement.

---

## Daily Streak

A streak increments every day you complete at least one recording session. Miss a day and it resets to zero.

- Streak is displayed in the sidebar on every page
- A ⚠️ warning appears if your streak is at risk (you drove yesterday but not today)
- Streak bonuses award bonus XP on top of your base lap XP (capped at +50 XP/day)

| Streak | Bonus XP |
|--------|----------|
| 2 days | +10 XP   |
| 5 days | +25 XP   |
| 10 days | +50 XP  |

---

## XP & Levels

You earn XP for everything you do in the app:

| Action | XP |
|--------|----|
| Recording a lap | +20 XP |
| Daily streak bonus | +5–50 XP |
| Completing a challenge | Varies (50–1000 XP) |
| Setting a personal best | +100 XP (via challenge) |

XP thresholds to level up increase progressively:

| Level | XP Required |
|-------|-------------|
| 1 | 100 |
| 2 | 250 |
| 3 | 450 |
| 4 | 700 |
| 5 | 1,000 |
| 10 | 5,000 |
| 15 | 11,000 |
| 20 | 26,000 |

---

## Titles

Your title unlocks based on your level and reflects your progression as a driver:

| Level | Title |
|-------|-------|
| 0–4 | 🏁 Rookie |
| 5–9 | ⚡ Hot Lap Hunter |
| 10–14 | 🎯 Apex Chaser |
| 15–19 | 🔪 Corner Surgeon |
| 20+ | 🏆 The Surgeon |

---

## Challenges

Challenges are split into four categories and tracked automatically from your session data. No manual input needed — the system detects completions and awards XP on your next session load.

### 🏁 Milestones
Progress-based challenges tied to raw activity:

| Challenge | Goal | XP |
|-----------|------|-----|
| First Blood | Complete your first lap | 50 |
| Getting Warmed Up | Record 10 laps total | 100 |
| Committed | Record 50 laps total | 300 |
| Track Regular | Record 100 laps total | 600 |
| Habit Forming | Complete 5 sessions | 150 |
| Dedicated Driver | Complete 20 sessions | 500 |
| On a Roll | 3-day streak | 100 |
| Weekly Warrior | 7-day streak | 300 |
| Obsessed | 30-day streak | 1,000 |

### ⚡ Performance
Improvement-based challenges:

| Challenge | Goal | XP |
|-----------|------|-----|
| Personal Best | Set a new best lap time | 100 |
| Always Improving | Set 5 personal bests | 400 |
| Corner Master | Improve apex speed by 10 km/h | 150 |

### ⏱️ Consistency
Technique-based challenges:

| Challenge | Goal | XP |
|-----------|------|-----|
| Smooth Operator | Get a clean corner (no issues) | 75 |
| Perfect Lap | Zero issues across all corners | 200 |
| Like Clockwork | 3 laps within 1 second of each other | 150 |

### 🧭 Exploration
Feature discovery challenges:

| Challenge | Goal | XP |
|-----------|------|-----|
| Data Scientist | Unlock the ML coach (5+ laps) | 200 |
| Record Keeper | Save a session to history | 50 |
| Globetrotter | Drive on 2 different tracks | 200 |

---

## Data Persistence

Your profile is saved to `player_profile.json` in the project directory. It stores:

- Current XP and level
- Streak and longest streak
- Last session date (for streak tracking)
- Total laps and sessions
- Challenge progress and completion status
- Last 100 XP log entries

The profile is loaded on dashboard startup and updated whenever you record or load a session.

---

## Implementation

The gamification system lives in `gamification.py` and is fully decoupled from the telemetry and ML layers:

```
gamification.py
├── PlayerProfile      — dataclass holding all player state
├── STATIC_CHALLENGES  — list of all challenge definitions
├── load_profile()     — deserializes from player_profile.json
├── save_profile()     — serializes to player_profile.json
├── award_xp()         — adds XP and handles level-ups
├── check_and_update_streak() — streak logic on session completion
├── update_challenges() — recomputes progress from current data
└── process_session()  — master hook called after any recording session
```
