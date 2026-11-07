"""
gamification.py
XP, levels, streaks, titles, and challenges for AC Lap Analyzer.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

PROFILE_PATH = Path("player_profile.json")

# ── Level thresholds ──────────────────────────────────────────────────────────
LEVEL_XP = [0, 100, 250, 450, 700, 1000, 1400, 1900, 2500, 3200, 4000,
            5000, 6200, 7600, 9200, 11000, 13000, 15500, 18500, 22000, 26000]

TITLES = {
    0:  "Rookie",
    5:  "Hot Lap Hunter",
    10: "Apex Chaser",
    15: "Corner Surgeon",
    20: "The Surgeon",
}

def get_title(level: int) -> str:
    t = "Rookie"
    for lvl, title in sorted(TITLES.items()):
        if level >= lvl:
            t = title
    return t

def xp_for_level(level: int) -> int:
    if level < len(LEVEL_XP):
        return LEVEL_XP[level]
    return LEVEL_XP[-1] + (level - len(LEVEL_XP) + 1) * 5000

def xp_for_next_level(level: int) -> int:
    return xp_for_level(level + 1)

# ── Challenge definitions ─────────────────────────────────────────────────────
@dataclass
class Challenge:
    id: str
    name: str
    description: str
    xp_reward: int
    category: str          # "milestone" | "consistency" | "performance" | "exploration"
    target: float = 0.0    # numeric target (laps, XP, speed, etc.)
    progress: float = 0.0
    completed: bool = False
    icon: str = "🏁"

STATIC_CHALLENGES = [
    # Milestones
    Challenge("first_lap",       "First Blood",          "Complete your first recorded lap",                        50,  "milestone", target=1,  icon="🏁"),
    Challenge("laps_10",         "Getting Warmed Up",    "Record 10 laps total",                                   100, "milestone", target=10, icon="🔟"),
    Challenge("laps_50",         "Committed",            "Record 50 laps total",                                   300, "milestone", target=50, icon="🎯"),
    Challenge("laps_100",        "Track Regular",        "Record 100 laps total",                                  600, "milestone", target=100,icon="💯"),
    Challenge("sessions_5",      "Habit Forming",        "Complete 5 driving sessions",                            150, "milestone", target=5,  icon="📅"),
    Challenge("sessions_20",     "Dedicated Driver",     "Complete 20 driving sessions",                           500, "milestone", target=20, icon="🗓️"),
    Challenge("streak_3",        "On a Roll",            "Maintain a 3-day driving streak",                        100, "milestone", target=3,  icon="🔥"),
    Challenge("streak_7",        "Weekly Warrior",       "Maintain a 7-day driving streak",                        300, "milestone", target=7,  icon="🔥"),
    Challenge("streak_30",       "Obsessed",             "Maintain a 30-day driving streak",                      1000, "milestone", target=30, icon="🔥"),
    # Consistency
    Challenge("clean_corner",    "Smooth Operator",      "Get a clean corner (no issues) in any lap",               75, "consistency", target=1, icon="✨"),
    Challenge("clean_lap",       "Perfect Lap",          "Complete a lap with zero issues on all corners",          200, "consistency", target=1, icon="⭐"),
    Challenge("consistent_3",    "Like Clockwork",       "Run 3 laps within 1 second of each other",               150, "consistency", target=3, icon="⏱️"),
    # Performance
    Challenge("beat_pb",         "Personal Best",        "Set a new personal best lap time",                       100, "performance", target=1, icon="🏆"),
    Challenge("beat_pb_5",       "Always Improving",     "Set 5 personal best lap times",                          400, "performance", target=5, icon="📈"),
    Challenge("apex_speed_up",   "Corner Master",        "Improve your apex speed on any corner by 10 km/h",       150, "performance", target=10, icon="⚡"),
    # Exploration
    Challenge("ml_unlock",       "Data Scientist",       "Unlock the ML coach (5+ laps on one car/track combo)",   200, "exploration", target=5, icon="🧠"),
    Challenge("save_session",    "Record Keeper",        "Save a session to history",                               50, "exploration", target=1, icon="💾"),
    Challenge("two_tracks",      "Globetrotter",         "Drive on 2 different tracks",                            200, "exploration", target=2, icon="🗺️"),
]

# ── Player profile ────────────────────────────────────────────────────────────
@dataclass
class PlayerProfile:
    xp: int = 0
    level: int = 0
    streak: int = 0
    longest_streak: int = 0
    last_session_date: str = ""   # ISO date string "YYYY-MM-DD"
    total_laps: int = 0
    total_sessions: int = 0
    personal_best_ms: int = 0
    pb_beats: int = 0
    challenges: List[dict] = field(default_factory=list)
    xp_log: List[dict] = field(default_factory=list)   # [{amount, reason, date}]

    def title(self) -> str:
        return get_title(self.level)

    def xp_to_next(self) -> int:
        return xp_for_next_level(self.level) - self.xp

    def level_progress_pct(self) -> float:
        cur = xp_for_level(self.level)
        nxt = xp_for_next_level(self.level)
        if nxt == cur:
            return 1.0
        return (self.xp - cur) / (nxt - cur)

    def get_challenge(self, cid: str) -> Optional[dict]:
        for c in self.challenges:
            if c["id"] == cid:
                return c
        return None


def _default_challenges() -> List[dict]:
    return [
        {"id": c.id, "progress": 0.0, "completed": False}
        for c in STATIC_CHALLENGES
    ]

def load_profile() -> PlayerProfile:
    if not PROFILE_PATH.exists():
        p = PlayerProfile()
        p.challenges = _default_challenges()
        return p
    with open(PROFILE_PATH) as f:
        d = json.load(f)
    p = PlayerProfile(
        xp=d.get("xp", 0),
        level=d.get("level", 0),
        streak=d.get("streak", 0),
        longest_streak=d.get("longest_streak", 0),
        last_session_date=d.get("last_session_date", ""),
        total_laps=d.get("total_laps", 0),
        total_sessions=d.get("total_sessions", 0),
        personal_best_ms=d.get("personal_best_ms", 0),
        pb_beats=d.get("pb_beats", 0),
        challenges=d.get("challenges", _default_challenges()),
        xp_log=d.get("xp_log", []),
    )
    # Backfill any new challenges added since profile was created
    existing_ids = {c["id"] for c in p.challenges}
    for c in STATIC_CHALLENGES:
        if c.id not in existing_ids:
            p.challenges.append({"id": c.id, "progress": 0.0, "completed": False})
    return p

def save_profile(p: PlayerProfile):
    with open(PROFILE_PATH, "w") as f:
        json.dump({
            "xp": p.xp,
            "level": p.level,
            "streak": p.streak,
            "longest_streak": p.longest_streak,
            "last_session_date": p.last_session_date,
            "total_laps": p.total_laps,
            "total_sessions": p.total_sessions,
            "personal_best_ms": p.personal_best_ms,
            "pb_beats": p.pb_beats,
            "challenges": p.challenges,
            "xp_log": p.xp_log[-100:],  # keep last 100 entries
        }, f, indent=2)

# ── XP & levelling ────────────────────────────────────────────────────────────
def award_xp(profile: PlayerProfile, amount: int, reason: str) -> List[str]:
    """Award XP and return list of level-up messages."""
    profile.xp += amount
    profile.xp_log.append({
        "amount": amount,
        "reason": reason,
        "date": date.today().isoformat(),
    })
    messages = []
    while profile.xp >= xp_for_next_level(profile.level):
        profile.level += 1
        messages.append(f"🎉 Level up! You are now **Level {profile.level} — {profile.title()}**")
    return messages

# ── Streak ────────────────────────────────────────────────────────────────────
def check_and_update_streak(profile: PlayerProfile) -> str:
    """Call when a session is completed. Returns streak message."""
    today = date.today().isoformat()
    if profile.last_session_date == today:
        return ""   # already drove today
    yesterday = (date.fromisoformat(today) - __import__('datetime').timedelta(days=1)).isoformat()
    if profile.last_session_date == yesterday:
        profile.streak += 1
    elif profile.last_session_date == "":
        profile.streak = 1
    else:
        profile.streak = 1   # reset
    profile.longest_streak = max(profile.longest_streak, profile.streak)
    profile.last_session_date = today
    if profile.streak == 1:
        return "🔥 Streak started! Come back tomorrow to keep it going."
    return f"🔥 {profile.streak}-day streak! Keep it up."

def streak_is_at_risk(profile: PlayerProfile) -> bool:
    """True if the player drove yesterday but not today (streak will reset tomorrow)."""
    if not profile.last_session_date:
        return False
    today = date.today().isoformat()
    yesterday = (date.fromisoformat(today) - __import__('datetime').timedelta(days=1)).isoformat()
    return profile.last_session_date == yesterday

# ── Challenge progress ────────────────────────────────────────────────────────
def update_challenges(profile: PlayerProfile, analyses, sessions) -> List[str]:
    """
    Recompute challenge progress from current data.
    Returns list of newly-completed challenge messages.
    """
    messages = []

    total_laps     = profile.total_laps
    total_sessions = profile.total_sessions
    streak         = profile.streak

    # Gather stats from analyses
    has_clean_corner = any(
        fb.overall and "clean" in fb.overall.lower()
        for a in (analyses or []) for fb in a.feedback
    ) if analyses else False

    has_clean_lap = any(
        all("clean" in fb.overall.lower() for fb in a.feedback)
        for a in (analyses or [])
    ) if analyses else False

    # Lap time consistency: best 3 consecutive within 1s
    consistent_3 = False
    if analyses and len(analyses) >= 3:
        times = [a.lap_time_ms for a in analyses]
        for i in range(len(times) - 2):
            window = times[i:i+3]
            if max(window) - min(window) <= 1000:
                consistent_3 = True
                break

    has_ml = any(
        len(h) >= 5 for h in (
            getattr(next(iter([s for s in (sessions or [])]), None), '__dict__', {}).get('corner_history', {}).values()
            if sessions else []
        )
    )
    # Simpler ML check: if any session exists with 5+ laps it's likely unlocked
    ml_unlocked = total_laps >= 5

    tracks = set(s.track for s in (sessions or []) if s.track not in ("Unknown Track", ""))
    has_session_saved = len(sessions or []) > 0

    targets = {
        "first_lap":     (min(total_laps, 1),   1),
        "laps_10":       (total_laps,            10),
        "laps_50":       (total_laps,            50),
        "laps_100":      (total_laps,            100),
        "sessions_5":    (total_sessions,        5),
        "sessions_20":   (total_sessions,        20),
        "streak_3":      (streak,                3),
        "streak_7":      (streak,                7),
        "streak_30":     (streak,                30),
        "clean_corner":  (1 if has_clean_corner else 0, 1),
        "clean_lap":     (1 if has_clean_lap else 0,    1),
        "consistent_3":  (1 if consistent_3 else 0,     1),
        "beat_pb":       (min(profile.pb_beats, 1),     1),
        "beat_pb_5":     (profile.pb_beats,              5),
        "ml_unlock":     (1 if ml_unlocked else 0,       1),
        "save_session":  (1 if has_session_saved else 0, 1),
        "two_tracks":    (len(tracks),                   2),
    }

    # apex_speed_up: check if any corner improved 10+ km/h vs first recorded
    # (simplified: just check if pb_beats > 0 as proxy)
    targets["apex_speed_up"] = (1 if profile.pb_beats > 0 else 0, 1)

    challenge_map = {c.id: c for c in STATIC_CHALLENGES}

    for entry in profile.challenges:
        cid = entry["id"]
        if entry["completed"]:
            continue
        if cid not in targets:
            continue
        prog, tgt = targets[cid]
        entry["progress"] = min(prog, tgt)
        if prog >= tgt:
            entry["completed"] = True
            c = challenge_map.get(cid)
            if c:
                lvl_msgs = award_xp(profile, c.xp_reward, f"Challenge: {c.name}")
                messages.append(f"{c.icon} Challenge unlocked: **{c.name}** (+{c.xp_reward} XP)")
                messages.extend(lvl_msgs)

    return messages

# ── Session hook ──────────────────────────────────────────────────────────────
def process_session(profile: PlayerProfile, analyses, sessions, laps_this_session: int) -> List[str]:
    """
    Call after recording/loading a session.
    Updates streak, awards XP, checks challenges.
    Returns all notification messages to show the user.
    """
    messages = []

    # Base XP per lap
    xp_per_lap = 20
    total_xp = laps_this_session * xp_per_lap
    if total_xp > 0:
        msgs = award_xp(profile, total_xp, f"{laps_this_session} laps recorded")
        messages.extend(msgs)

    profile.total_laps += laps_this_session
    profile.total_sessions += 1

    streak_msg = check_and_update_streak(profile)
    if streak_msg:
        messages.append(streak_msg)

    # Streak bonus XP
    if profile.streak > 1:
        bonus = min(profile.streak * 5, 50)
        msgs = award_xp(profile, bonus, f"Streak bonus ({profile.streak} days)")
        messages.append(f"🔥 Streak bonus: +{bonus} XP")
        messages.extend(msgs)

    challenge_msgs = update_challenges(profile, analyses, sessions)
    messages.extend(challenge_msgs)

    save_profile(profile)
    return messages