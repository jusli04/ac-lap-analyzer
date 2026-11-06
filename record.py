"""
record.py
Standalone CLI recorder — run this while AC is open to capture laps.
Saves to saved_laps.pkl which the dashboard can then load.

Usage:
    python record.py
    python record.py --duration 600   (record for up to 10 minutes)
"""

import argparse
import pickle
from pathlib import Path
from ac_telemetry import LapRecorder
from analyzer import analyze_lap, print_analysis, format_lap_time
from ml_coach import MLCoachState, update_history
from history import build_session_record, save_session, total_laps_for_combo

def main():
    parser = argparse.ArgumentParser(description="AC Lap Telemetry Recorder")
    parser.add_argument("--duration", type=int, default=300,
                        help="Max recording duration in seconds (default: 300)")
    parser.add_argument("--hz", type=int, default=20,
                        help="Sample rate in Hz (default: 20)")
    parser.add_argument("--out", type=str, default="saved_laps.pkl",
                        help="Output file path")
    args = parser.parse_args()

    print("="*60)
    print("  Assetto Corsa Lap Analyzer — Recorder")
    print("="*60)
    print(f"  Sample rate : {args.hz} Hz")
    print(f"  Max duration: {args.duration}s")
    print(f"  Output file : {args.out}")
    print("="*60)
    print("\nMake sure AC is running and you are IN a session (not the menu).\n")

    recorder = LapRecorder(sample_rate_hz=args.hz)
    laps = recorder.record(duration_seconds=args.duration)

    if not laps:
        print("No laps recorded.")
        return

    # Save laps
    with open(args.out, "wb") as f:
        pickle.dump(laps, f)
    print(f"\nSaved {len(laps)} laps to {args.out}")

    # Quick analysis summary
    print("\n" + "="*60)
    print("  QUICK ANALYSIS SUMMARY")
    print("="*60)

    best_lap  = None
    best_time = float("inf")
    analyses  = []
    ml_state  = MLCoachState()

    for i, lap in enumerate(laps):
        # Use AC's own iLastTime captured at lap boundary — most accurate source
        lap_time = recorder.lap_times_ms[i] if i < len(recorder.lap_times_ms) else 0
        analysis = analyze_lap(lap, i + 1, ref_samples=best_lap, lap_time_ms=lap_time)
        analyses.append(analysis)
        ml_state = update_history(ml_state, analysis, lap)
        print_analysis(analysis)

        if lap_time < best_time:
            best_time = lap_time
            best_lap  = lap

    print(f"\n🏆 Best lap: {format_lap_time(best_time)}")

    # Offer to save session to history
    print("\n" + "="*60)
    save_choice = input("Save this session to history? (y/n): ").strip().lower()
    if save_choice == "y":
        track = input("Track name (press Enter to skip): ").strip() or "Unknown Track"
        car   = input("Car (press Enter to skip): ").strip() or "Unknown Car"
        record = build_session_record(analyses, ml_state, track=track, car=car)
        save_session(record)
        total_cross = total_laps_for_combo(track, car)
        ml_min = ml_state.min_laps_for_model
        print(f"✅ Session saved — {record.laps_driven} laps, best {record.best_lap_str()}")
        if total_cross >= ml_min:
            print(f"🧠 ML Coach unlocked! {total_cross} laps on record for {car} at {track}.")
        else:
            print(f"🧠 ML Coach: {total_cross}/{ml_min} laps recorded for {car} at {track}. "
                  f"Record {ml_min - total_cross} more to unlock insights.")

    print("\nRun `streamlit run dashboard.py` to view the full visual dashboard.")


if __name__ == "__main__":
    main()