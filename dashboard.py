"""
dashboard.py
Streamlit dashboard for AC Lap Analyzer.
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import pickle
from pathlib import Path
from analyzer import analyze_lap, format_lap_time, LapAnalysis
from ac_telemetry import TelemetrySample, LapRecorder
from ml_coach import MLCoachState, update_history, analyze_lap_ml, compute_corner_stats
from history import (load_sessions, save_session, build_session_record,
                     progression_data, corner_history_heatmap, personal_bests,
                     load_ml_state_for_combo, total_laps_for_combo)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AC Lap Analyzer",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        margin: 4px 0;
    }
    .issue-badge {
        background: #ff4b4b22;
        border-left: 3px solid #ff4b4b;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        margin: 4px 0;
        font-size: 0.9em;
    }
    .positive-badge {
        background: #00cc6622;
        border-left: 3px solid #00cc66;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        margin: 4px 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "laps" not in st.session_state:
    st.session_state.laps = []
if "analyses" not in st.session_state:
    st.session_state.analyses = []
if "ml_state" not in st.session_state:
    st.session_state.ml_state = MLCoachState()
if "ml_insights" not in st.session_state:
    st.session_state.ml_insights = []


# ── Helpers ───────────────────────────────────────────────────────────────────
SAVE_PATH = Path("saved_laps.pkl")

def save_laps(laps):
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(laps, f)

def load_laps():
    if SAVE_PATH.exists():
        with open(SAVE_PATH, "rb") as f:
            return pickle.load(f)
    return []

def samples_to_df(samples):
    return pd.DataFrame([{
        "lap_time_s":    s.lap_time_ms / 1000,
        "track_pos":     s.track_position,
        "speed_kmh":     s.speed_kmh,
        "throttle":      s.throttle,
        "brake":         s.brake,
        "steer_angle":   s.steer_angle,
        "gear":          s.gear,
        "rpms":          s.rpms,
        "g_lat":         s.g_lat,
        "g_lon":         s.g_lon,
        "x":             s.x,
        "z":             s.z,
    } for s in samples])


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏎️ AC Lap Analyzer")
st.sidebar.markdown("---")

page = st.sidebar.radio("View", ["📊 Dashboard", "🔴 Live Record", "📁 Load Saved", "📈 History"])

# Load saved laps
if st.sidebar.button("Load Saved Laps") or not st.session_state.laps:
    loaded = load_laps()
    if loaded:
        st.session_state.laps = loaded
        st.session_state.analyses = []
        st.session_state.ml_state = MLCoachState()
        st.session_state.ml_insights = []
        best_lap = None
        for i, lap in enumerate(loaded):
            ref = best_lap
            analysis = analyze_lap(lap, i + 1, ref_samples=ref)
            st.session_state.analyses.append(analysis)
            # Update ML state with this lap BEFORE running insights (so it learns from it)
            st.session_state.ml_state = update_history(st.session_state.ml_state, analysis, lap)
            ml = analyze_lap_ml(st.session_state.ml_state, analysis, lap)
            st.session_state.ml_insights.append(ml)
            if best_lap is None or lap[-1].lap_time_ms < best_lap[-1].lap_time_ms:
                best_lap = lap
        st.sidebar.success(f"Loaded {len(loaded)} laps")


# ── Dashboard page ────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("🏎️ Lap Analysis Dashboard")

    if not st.session_state.laps:
        st.info("No laps recorded yet. Go to **Live Record** to capture telemetry, or **Load Saved** to load previous sessions.")
        st.stop()

    analyses = st.session_state.analyses
    laps     = st.session_state.laps

    # Lap selector
    lap_options = [f"Lap {a.lap_number} — {format_lap_time(a.lap_time_ms)}" for a in analyses]
    selected_label = st.selectbox("Select Lap", lap_options)
    selected_idx   = lap_options.index(selected_label)
    analysis = analyses[selected_idx]
    samples  = laps[selected_idx]
    df       = samples_to_df(samples)

    # ── Top metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lap Time",       format_lap_time(analysis.lap_time_ms))
    col2.metric("Corners Found",  len(analysis.corners))
    col3.metric("Avg Apex Speed", f"{analysis.avg_apex_speed:.1f} km/h")
    col4.metric("Issues Found",   analysis.total_issues)

    st.markdown("---")

    # ── Speed / throttle / brake trace ──
    st.subheader("Lap Trace")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["track_pos"], y=df["speed_kmh"],
                             name="Speed (km/h)", line=dict(color="#00aaff", width=2)))
    fig.add_trace(go.Scatter(x=df["track_pos"], y=df["throttle"] * 100,
                             name="Throttle %", line=dict(color="#00cc66", width=1.5)))
    fig.add_trace(go.Scatter(x=df["track_pos"], y=df["brake"] * 100,
                             name="Brake %", line=dict(color="#ff4b4b", width=1.5)))

    # Mark corner apexes
    for corner in analysis.corners:
        apex_pos = corner.track_pos_apex
        fig.add_vline(x=apex_pos, line_dash="dot", line_color="yellow",
                      annotation_text=f"C{analysis.corners.index(corner)+1}",
                      annotation_position="top")

    fig.update_layout(
        xaxis_title="Track Position (0-1)",
        yaxis_title="Value",
        height=350,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Track map ──
    col_map, col_coaching = st.columns([1, 1])

    with col_map:
        st.subheader("Track Map")
        fig_map = go.Figure()
        fig_map.add_trace(go.Scatter(
            x=df["x"], y=df["z"],
            mode="markers",
            marker=dict(
                color=df["speed_kmh"],
                colorscale="RdYlGn",
                size=3,
                cmin=df["speed_kmh"].min(),
                cmax=df["speed_kmh"].max(),
                colorbar=dict(title="km/h"),
            ),
            name="Speed",
        ))
        # Mark apexes
        for i, corner in enumerate(analysis.corners):
            apex_s = samples[corner.apex_idx]
            fig_map.add_trace(go.Scatter(
                x=[apex_s.x], y=[apex_s.z],
                mode="markers+text",
                marker=dict(size=10, color="yellow"),
                text=[f"C{i+1}"],
                textposition="top center",
                showlegend=False,
            ))
        fig_map.update_layout(
            height=350,
            template="plotly_dark",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, scaleanchor="x"),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_coaching:
        st.subheader("Corner-by-Corner Coaching")
        for fb in analysis.feedback:
            with st.expander(f"Corner {fb.corner_number}  —  {fb.overall}"):
                if fb.issues:
                    st.markdown("**Issues:**")
                    for issue in fb.issues:
                        st.markdown(f'<div class="issue-badge">⚠ {issue}</div>',
                                    unsafe_allow_html=True)
                if fb.positives:
                    st.markdown("**Positives:**")
                    for pos in fb.positives:
                        st.markdown(f'<div class="positive-badge">✓ {pos}</div>',
                                    unsafe_allow_html=True)

    # ── Lap comparison ──
    if len(analyses) > 1:
        st.markdown("---")
        st.subheader("Lap Comparison")
        compare_data = pd.DataFrame([{
            "Lap":        f"Lap {a.lap_number}",
            "Time (s)":   a.lap_time_ms / 1000,
            "Avg Apex Speed": a.avg_apex_speed,
            "Issues":     a.total_issues,
        } for a in analyses])

        fig_comp = px.bar(compare_data, x="Lap", y="Time (s)",
                          color="Issues", color_continuous_scale="RdYlGn_r",
                          template="plotly_dark")
        fig_comp.update_layout(height=280)
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── ML Insights ──
    st.markdown("---")
    st.subheader("🧠 ML Coach Insights")

    ml_insights = st.session_state.ml_insights
    ml_state    = st.session_state.ml_state
    total_laps  = max((len(h) for h in ml_state.corner_history.values()), default=0)
    needed      = ml_state.min_laps_for_model

    # Try to load a richer cross-session ML state if a track+car is known for this session
    # (only relevant after the user has saved sessions to history)
    _cross_track = st.session_state.get("current_track", "")
    _cross_car   = st.session_state.get("current_car", "")
    if _cross_track and _cross_car:
        cross_state = load_ml_state_for_combo(_cross_track, _cross_car)
        cross_laps  = max((len(h) for h in cross_state.corner_history.values()), default=0)
        if cross_laps > total_laps:
            ml_state   = cross_state
            total_laps = cross_laps

    if total_laps < needed:
        laps_left = needed - total_laps
        # Show cross-session count if we know the combo
        if _cross_track and _cross_car:
            st.info(
                f"ML Coach needs at least {needed} laps across all your sessions at "
                f"**{_cross_track}** in the **{_cross_car}**. "
                f"You have **{total_laps}** so far — record {laps_left} more to unlock this."
            )
        else:
            st.info(
                f"ML Coach needs at least {needed} laps to start working. "
                f"Record {laps_left} more lap{'s' if laps_left != 1 else ''} to unlock this."
            )
    else:
        ml = ml_insights[selected_idx] if selected_idx < len(ml_insights) else None

        if ml:
            # Summary callout
            st.markdown(f"**{ml.summary}**")
            st.markdown("---")

            col_ml1, col_ml2 = st.columns(2)

            with col_ml1:
                # Corner importance bar chart
                st.markdown("**Corner Importance** *(how much each corner affects your lap time)*")
                if ml.corner_importances:
                    imp_df = pd.DataFrame(ml.corner_importances, columns=["Corner", "Importance"])
                    imp_df["Corner"] = imp_df["Corner"].apply(lambda c: f"C{c}")
                    imp_df = imp_df.sort_values("Importance", ascending=True)
                    fig_imp = go.Figure(go.Bar(
                        x=imp_df["Importance"],
                        y=imp_df["Corner"],
                        orientation="h",
                        marker_color=[
                            f"rgba({int(255*(1-v))}, {int(200*v)}, 80, 0.85)"
                            for v in imp_df["Importance"]
                        ],
                    ))
                    fig_imp.update_layout(
                        height=300,
                        template="plotly_dark",
                        xaxis_title="Importance (0-1)",
                        margin=dict(l=0, r=0, t=10, b=0),
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

            with col_ml2:
                # Time gains available
                st.markdown("**Time Available** *(seconds to gain if you match your personal bests)*")
                if ml.gains_available:
                    gains_df = pd.DataFrame(ml.gains_available, columns=["Corner", "Gain (s)"])
                    gains_df["Corner"] = gains_df["Corner"].apply(lambda c: f"C{c}")
                    fig_gains = go.Figure(go.Bar(
                        x=gains_df["Corner"],
                        y=gains_df["Gain (s)"],
                        marker_color="#00aaff",
                        text=[f"{g:.3f}s" for g in gains_df["Gain (s)"]],
                        textposition="outside",
                    ))
                    fig_gains.update_layout(
                        height=300,
                        template="plotly_dark",
                        yaxis_title="Seconds",
                        margin=dict(l=0, r=0, t=10, b=30),
                    )
                    st.plotly_chart(fig_gains, use_container_width=True)
                else:
                    st.info("No gains detected — you're hitting your personal bests!")

            # Anomalies
            if ml.anomalous_corners:
                st.markdown("**Anomalies this lap:**")
                for corner_num, desc in ml.anomalous_corners:
                    if "went wrong" in desc:
                        st.markdown(
                            f'<div class="issue-badge">⚠ Corner {corner_num}: {desc}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="positive-badge">✅ Corner {corner_num}: {desc}</div>',
                            unsafe_allow_html=True
                        )

            # Prediction vs actual
            st.markdown("---")
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            pred_col1.metric("Actual Lap Time",    format_lap_time(ml.actual_lap_time_ms))
            pred_col2.metric("Predicted Lap Time", format_lap_time(ml.predicted_lap_time_ms))
            delta_s = ml.prediction_error_ms / 1000
            pred_col3.metric("Delta", f"{delta_s:+.3f}s",
                              delta=f"{-delta_s:.3f}s faster" if delta_s > 0 else f"{abs(delta_s):.3f}s faster than predicted")

        else:
            st.info("ML analysis not available for this lap yet.")


# ── Live record page ──────────────────────────────────────────────────────────
elif page == "🔴 Live Record":
    st.title("🔴 Live Telemetry Recording")
    st.info("Start Assetto Corsa, load a session, and click Record below.")

    duration = st.slider("Max recording duration (seconds)", 60, 600, 300, step=30)

    if st.button("▶ Start Recording"):
        with st.spinner("Recording... Drive your laps! Press Ctrl+C in the terminal to stop early."):
            recorder = LapRecorder(sample_rate_hz=20)
            laps = recorder.record(duration_seconds=duration)

        if laps:
            st.session_state.laps = laps
            save_laps(laps)

            best_lap = None
            analyses = []
            for i, lap in enumerate(laps):
                analysis = analyze_lap(lap, i + 1, ref_samples=best_lap)
                analyses.append(analysis)
                if best_lap is None or lap[-1].lap_time_ms < best_lap[-1].lap_time_ms:
                    best_lap = lap
            st.session_state.analyses = analyses

            st.success(f"✅ Recorded {len(laps)} laps. Go to Dashboard to view analysis.")
        else:
            st.error("No laps recorded. Make sure AC is running and you completed at least one lap.")


# ── Load saved page ───────────────────────────────────────────────────────────
elif page == "📁 Load Saved":
    st.title("📁 Load Saved Session")

    if SAVE_PATH.exists():
        st.success(f"Found saved session: {SAVE_PATH}")
        if st.button("Load"):
            loaded = load_laps()
            st.session_state.laps = loaded
            st.session_state.ml_state = MLCoachState()
            st.session_state.ml_insights = []
            best_lap = None
            analyses = []
            for i, lap in enumerate(loaded):
                analysis = analyze_lap(lap, i + 1, ref_samples=best_lap)
                analyses.append(analysis)
                st.session_state.ml_state = update_history(st.session_state.ml_state, analysis, lap)
                ml = analyze_lap_ml(st.session_state.ml_state, analysis, lap)
                st.session_state.ml_insights.append(ml)
                if best_lap is None or lap[-1].lap_time_ms < best_lap[-1].lap_time_ms:
                    best_lap = lap
            st.session_state.analyses = analyses
            st.success(f"Loaded {len(loaded)} laps. Go to Dashboard.")
    else:
        st.warning("No saved session found. Record a session first.")


# ── History page ──────────────────────────────────────────────────────────────
elif page == "📈 History":
    st.title("📈 Session History")

    sessions = load_sessions()

    if not sessions:
        st.info("No session history yet. Complete a recording session and save it to start building your history.")
        st.stop()

    # ── Personal bests banner ──
    pb = personal_bests(sessions)
    pb_col1, pb_col2, pb_col3, pb_col4 = st.columns(4)
    pb_col1.metric("All-Time Best Lap",  format_lap_time(pb["best_lap_ms"]))
    pb_col2.metric("Best Lap Set",       pb["best_lap_session"])
    pb_col3.metric("Total Sessions",     pb["total_sessions"])
    pb_col4.metric("Total Laps Driven",  pb["total_laps"])

    st.markdown("---")

    # ── Progression chart ──
    st.subheader("Lap Time Progression")
    prog = progression_data(sessions)

    fig_prog = go.Figure()
    fig_prog.add_trace(go.Scatter(
        x=prog["dates"], y=prog["best_laps_s"],
        mode="lines+markers",
        name="Best Lap",
        line=dict(color="#00aaff", width=2),
        marker=dict(size=8),
        customdata=list(zip(prog["tracks"], prog["laps_driven"])),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Best: %{y:.3f}s<br>"
            "Track: %{customdata[0]}<br>"
            "Laps: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig_prog.add_trace(go.Scatter(
        x=prog["dates"], y=prog["avg_laps_s"],
        mode="lines+markers",
        name="Avg Lap",
        line=dict(color="#888888", width=1.5, dash="dot"),
        marker=dict(size=6),
    ))
    fig_prog.update_layout(
        height=320,
        template="plotly_dark",
        yaxis_title="Lap Time (s)",
        xaxis_title="Session",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_prog, use_container_width=True)

    # ── Corner consistency heatmap ──
    st.subheader("Corner Speed Heatmap *(avg apex speed per session)*")
    hm = corner_history_heatmap(sessions)

    if hm["matrix"] and any(any(v is not None for v in row) for row in hm["matrix"]):
        # Replace None with 0 for display
        matrix_clean = [
            [v if v is not None else 0 for v in row]
            for row in hm["matrix"]
        ]
        fig_hm = go.Figure(go.Heatmap(
            z=matrix_clean,
            x=hm["corner_labels"],
            y=hm["session_labels"],
            colorscale="RdYlGn",
            colorbar=dict(title="km/h"),
            hoverongaps=False,
            hovertemplate="Session: %{y}<br>%{x}: %{z:.1f} km/h<extra></extra>",
        ))
        fig_hm.update_layout(
            height=max(200, len(sessions) * 40 + 80),
            template="plotly_dark",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Corner",
            yaxis_title="Session",
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Green = fast apex speed. Red = slow. Track your corner-by-corner improvement over time.")
    else:
        st.info("Corner heatmap needs sessions with ML data (5+ laps each).")

    st.markdown("---")

    # ── Track leaderboard ──
    st.subheader("🏆 Track Leaderboard *(your best lap per car)*")

    # Build: track → {car → best_lap_ms, sessions}
    from collections import defaultdict
    leaderboard: dict = defaultdict(dict)
    for s in sessions:
        track = s.track
        car   = s.car
        if track not in leaderboard or car not in leaderboard[track]:
            leaderboard[track][car] = s.best_lap_ms
        else:
            leaderboard[track][car] = min(leaderboard[track][car], s.best_lap_ms)

    all_lb_tracks = sorted(leaderboard.keys())
    selected_lb_track = st.selectbox("Select track", all_lb_tracks, key="lb_track")

    if selected_lb_track:
        car_times = leaderboard[selected_lb_track]
        # Sort by lap time ascending
        sorted_cars = sorted(car_times.items(), key=lambda x: x[1])
        fastest_time = sorted_cars[0][1]

        lb_rows = []
        for rank, (car, best_ms) in enumerate(sorted_cars, 1):
            delta_ms = best_ms - fastest_time
            delta_str = "—" if delta_ms == 0 else f"+{delta_ms/1000:.3f}s"
            lb_rows.append({
                "Rank": f"{'🥇' if rank==1 else '🥈' if rank==2 else '🥉' if rank==3 else str(rank)}",
                "Car": car,
                "Best Lap": format_lap_time(best_ms),
                "Gap to Fastest": delta_str,
            })

        lb_df = pd.DataFrame(lb_rows)
        st.dataframe(
            lb_df,
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart — only if more than one car
        if len(sorted_cars) > 1:
            fig_lb = go.Figure(go.Bar(
                x=[r["Car"] for r in lb_rows],
                y=[t / 1000 for _, t in sorted_cars],
                marker_color=["#ffd700" if i == 0 else "#00aaff" for i in range(len(sorted_cars))],
                text=[r["Best Lap"] for r in lb_rows],
                textposition="outside",
            ))
            fig_lb.update_layout(
                height=300,
                template="plotly_dark",
                yaxis_title="Best Lap (s)",
                yaxis=dict(range=[
                    min(t for _, t in sorted_cars) * 0.98 / 1000,
                    max(t for _, t in sorted_cars) * 1.01 / 1000,
                ]),
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_lb, use_container_width=True)

    st.markdown("---")

    # ── Session cards ──
    st.subheader("Session Log")
    for s in reversed(sessions):  # newest first
        with st.expander(f"**{s.date}** — {s.track}  |  Best: {s.best_lap_str()}  |  {s.laps_driven} laps"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Lap",   s.best_lap_str())
            c2.metric("Avg Lap",    s.avg_lap_str())
            c3.metric("Laps",       s.laps_driven)
            c4.metric("Car",        s.car if s.car != "Unknown Car" else "—")
            st.markdown(f"**Key Insight:** {s.top_insight}")

    # ── Cross-session ML Coach ──
    st.markdown("---")
    st.subheader("🧠 ML Coach — Cross-Session Insights")
    st.caption("Select a car + track combination to see ML insights pooled across all your saved sessions.")

    all_tracks = sorted(set(s.track for s in sessions if s.track not in ("Unknown Track",)))
    all_cars   = sorted(set(s.car   for s in sessions if s.car   not in ("Unknown Car",)))

    if not all_tracks or not all_cars:
        st.info("Save at least one session with a track and car name to unlock cross-session ML insights.")
    else:
        ml_track = st.selectbox("Track", all_tracks, key="hist_ml_track")
        ml_car   = st.selectbox("Car",   all_cars,   key="hist_ml_car")

        cross_state = load_ml_state_for_combo(ml_track, ml_car)
        cross_laps  = max((len(h) for h in cross_state.corner_history.values()), default=0)
        needed      = cross_state.min_laps_for_model

        matched_sessions = [s for s in sessions
                            if s.track.strip().lower() == ml_track.strip().lower()
                            and s.car.strip().lower() == ml_car.strip().lower()]

        st.markdown(
            f"**{len(matched_sessions)} session(s)** found for **{ml_car}** at **{ml_track}** "
            f"— **{cross_laps} laps** in ML pool."
        )

        if cross_laps < needed:
            laps_left = needed - cross_laps
            st.info(
                f"Need {laps_left} more lap{'s' if laps_left != 1 else ''} to unlock ML insights "
                f"for this combination. Keep recording!"
            )
        else:
            from ml_coach import compute_corner_stats
            corner_stats = compute_corner_stats(cross_state)
            if corner_stats:
                st.markdown("**Corner Importance** *(pooled across all saved sessions)*")
                import pandas as pd
                imp_df = pd.DataFrame([
                    {"Corner": f"C{cs.corner_number}",
                     "Importance": cs.importance_score,
                     "Avg Apex (km/h)": cs.avg_apex_speed,
                     "Gain per km/h (s)": cs.lap_time_gain_per_kmh}
                    for cs in corner_stats
                ])

                fig_imp = go.Figure(go.Bar(
                    x=imp_df["Importance"],
                    y=imp_df["Corner"],
                    orientation="h",
                    marker_color=[
                        f"rgba({int(255*(1-v))}, {int(200*v)}, 80, 0.85)"
                        for v in imp_df["Importance"]
                    ],
                    customdata=imp_df[["Avg Apex (km/h)", "Gain per km/h (s)"]].values,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Importance: %{x:.2f}<br>"
                        "Avg apex: %{customdata[0]:.1f} km/h<br>"
                        "Worth: %{customdata[1]:.3f}s per km/h<extra></extra>"
                    ),
                ))
                fig_imp.update_layout(
                    height=max(200, len(corner_stats) * 30 + 60),
                    template="plotly_dark",
                    xaxis_title="Importance (0–1)",
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown("**Corner detail:**")
                st.dataframe(imp_df.style.format({
                    "Importance": "{:.2f}",
                    "Avg Apex (km/h)": "{:.1f}",
                    "Gain per km/h (s)": "{:.3f}",
                }), use_container_width=True)

    # ── Save current session button ──
    st.markdown("---")
    st.subheader("Save Current Session to History")
    if st.session_state.analyses:
        track_input = st.text_input("Track name (optional)", placeholder="e.g. Monza, Silverstone")
        car_input   = st.text_input("Car (optional)",        placeholder="e.g. Ferrari 488 GT3")
        if st.button("💾 Save This Session"):
            track_val = track_input or "Unknown Track"
            car_val   = car_input   or "Unknown Car"
            record = build_session_record(
                analyses  = st.session_state.analyses,
                ml_state  = st.session_state.ml_state,
                track     = track_val,
                car       = car_val,
            )
            save_session(record)
            # Remember this combo for the Dashboard ML section
            st.session_state["current_track"] = track_val
            st.session_state["current_car"]   = car_val
            st.success(f"Session saved! Best lap: {record.best_lap_str()} over {record.laps_driven} laps.")
            st.rerun()
    else:
        st.info("No session loaded. Record or load laps first, then come back here to save.")