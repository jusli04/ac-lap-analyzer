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
from gamification import (
    load_profile, save_profile, process_session, update_challenges,
    award_xp, check_and_update_streak, streak_is_at_risk,
    STATIC_CHALLENGES, PlayerProfile,
)

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
if "profile" not in st.session_state:
    st.session_state.profile = load_profile()
if "gami_messages" not in st.session_state:
    st.session_state.gami_messages = []


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

page = st.sidebar.radio("View", ["📊 Dashboard", "🔴 Live Record", "📁 Load Saved", "📈 History", "🏆 Challenges"])

# ── Gamification sidebar widget ───────────────────────────────────────────────
st.sidebar.markdown("---")
_p = st.session_state.profile
_streak_color = "#ff6b35" if _p.streak > 0 else "#555"
_at_risk = streak_is_at_risk(_p)
_streak_label = f"🔥 {_p.streak}-day streak" if _p.streak > 0 else "🔥 No streak yet"
if _at_risk:
    _streak_label += " ⚠️"
st.sidebar.markdown(f'''<div style="background:#1a1a2e;border-radius:8px;padding:12px 14px;margin-bottom:8px">
<div style="font-size:1.1em;font-weight:bold;color:{_streak_color}">{_streak_label}</div>
<div style="font-size:0.9em;color:#aaa;margin-top:2px">⚡ Level {_p.level} — {_p.title()}</div>
<div style="margin-top:6px">
<div style="background:#333;border-radius:4px;height:8px">
<div style="background:linear-gradient(90deg,#f7971e,#ffd200);width:{min(int(_p.level_progress_pct()*100),100)}%;height:8px;border-radius:4px"></div>
</div>
<div style="font-size:0.75em;color:#888;margin-top:2px">{_p.xp} XP — {_p.xp_to_next()} to Level {_p.level+1}</div>
</div>
</div>''', unsafe_allow_html=True)

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

    # selected corner state
    if "selected_corner" not in st.session_state:
        st.session_state.selected_corner = None

    with col_map:
        st.subheader("Track Map")
        sel = st.session_state.selected_corner
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
                colorbar=dict(title="km/h", x=-0.12, xanchor="right", thickness=12, len=0.7),
            ),
            name="Speed",
            showlegend=False,
        ))
        sf_s = max(samples, key=lambda s: s.track_position)
        fig_map.add_trace(go.Scatter(
            x=[sf_s.x], y=[sf_s.z],
            mode="markers+text",
            marker=dict(size=18, color="white", symbol="square"),
            text=["S/F"],
            textposition="top center",
            showlegend=False,
        ))
        for i, corner in enumerate(analysis.corners):
            apex_s = samples[corner.apex_idx]
            is_sel = (sel == i)
            fig_map.add_trace(go.Scatter(
                x=[apex_s.x], y=[apex_s.z],
                mode="markers+text",
                marker=dict(size=14 if is_sel else 8, color="cyan" if is_sel else "yellow", symbol="star" if is_sel else "circle"),
                text=[f"C{i+1}"],
                textposition="top center",
                showlegend=False,
                name=f"C{i+1}",
            ))
            if is_sel:
                entry_s = samples[corner.entry_idx]
                exit_s  = samples[corner.exit_idx]
                fig_map.add_trace(go.Scatter(
                    x=[entry_s.x, apex_s.x, exit_s.x],
                    y=[entry_s.z, apex_s.z, exit_s.z],
                    mode="lines",
                    line=dict(color="cyan", width=3, dash="dot"),
                    showlegend=False,
                    name="corner zone",
                ))
                if corner.brake_start_idx is not None:
                    brake_s = samples[corner.brake_start_idx]
                    fig_map.add_trace(go.Scatter(
                        x=[brake_s.x], y=[brake_s.z],
                        mode="markers+text",
                        marker=dict(size=16, color="red", symbol="triangle-down"),
                        text=["BRAKE"],
                        textposition="bottom center",
                        showlegend=False,
                        name="brake point",
                    ))
        fig_map.update_layout(
            height=400,
            template="plotly_dark",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, scaleanchor="x"),
            xaxis_autorange="reversed",
            margin=dict(l=60, r=10, t=10, b=10),
            modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0.3)"),
        )
        st.plotly_chart(fig_map, use_container_width=True)
        if sel is not None:
            fb = analysis.feedback[sel]
            tips = fb.issues + fb.positives
            if tips:
                tip_html = "".join(
                    f'<div style="background:#1a1a2e;border-left:3px solid {"#ff4b4b" if t in fb.issues else "#00cc66"};padding:6px 10px;margin:3px 0;border-radius:0 4px 4px 0;font-size:0.85em">{"⚠" if t in fb.issues else "✓"} {t}</div>'
                    for t in tips
                )
                st.markdown(f'<div style="margin-top:8px"><b>Corner {sel+1} tips:</b>{tip_html}</div>', unsafe_allow_html=True)
            else:
                st.success(f"Corner {sel+1}: Clean! Nothing to fix here.")

    with col_coaching:
        st.subheader("Corner-by-Corner Coaching")
        st.caption("Click a corner to highlight it on the map")
        for i, fb in enumerate(analysis.feedback):
            border = "border:2px solid cyan;" if st.session_state.selected_corner == i else ""
            label = f"Corner {fb.corner_number}  —  {fb.overall}"
            with st.expander(label, expanded=(st.session_state.selected_corner == i)):
                if st.button(f"Show on map", key=f"sel_corner_{i}_{selected_idx}"):
                    st.session_state.selected_corner = i if st.session_state.selected_corner != i else None
                    st.rerun()
                if fb.issues:
                    st.markdown("**Issues:**")
                    for issue in fb.issues:
                        st.markdown(f'<div class="issue-badge">⚠ {issue}</div>', unsafe_allow_html=True)
                if fb.positives:
                    st.markdown("**Positives:**")
                    for pos in fb.positives:
                        st.markdown(f'<div class="positive-badge">✓ {pos}</div>', unsafe_allow_html=True)

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

    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "recorder_laps" not in st.session_state:
        st.session_state.recorder_laps = []

    col_start, col_stop = st.columns([1, 1])
    with col_start:
        start_clicked = st.button("▶ Start Recording", disabled=st.session_state.recording)
    with col_stop:
        stop_clicked = st.button("⏹ Stop Recording", disabled=not st.session_state.recording)

    if start_clicked:
        import threading
        st.session_state.recording = True
        st.session_state.recorder_laps = []
        stop_evt = threading.Event()
        st.session_state["stop_event"] = stop_evt
        recorder = LapRecorder(sample_rate_hz=20)
        st.session_state["recorder_obj"] = recorder
        def _run():
            laps = recorder.record(duration_seconds=duration, stop_event=stop_evt)
            st.session_state.recorder_laps = laps or []
            st.session_state.recording = False
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        st.session_state["record_thread"] = t
        st.rerun()

    if stop_clicked:
        evt = st.session_state.get("stop_event")
        if evt:
            evt.set()
        st.session_state.recording = False
        st.rerun()

    if st.session_state.recording:
        st.info("🔴 Recording in progress - drive your laps, then click Stop Recording.")
        st.stop()

    laps = st.session_state.get("recorder_laps", [])
    if not laps:
        pass

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
        st.session_state.recorder_laps = []
        # Gamification
        sessions = load_sessions()
        gami_msgs = process_session(
            st.session_state.profile,
            st.session_state.analyses,
            sessions,
            len(laps),
        )
        st.session_state.gami_messages = gami_msgs
        st.success(f"✅ Recorded {len(laps)} laps. Go to Dashboard to view analysis.")
        for msg in gami_msgs:
            st.info(msg)


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


# ── Challenges page ────────────────────────────────────────────────────────────────────────────────
elif page == "🏆 Challenges":
    p = st.session_state.profile
    sessions = load_sessions()

    # refresh challenge progress
    update_challenges(p, st.session_state.analyses, sessions)
    save_profile(p)

    st.title("🏆 Challenges & Profile")

    # ── Profile card ──
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.metric("Level", f"{p.level} — {p.title()}")
    col_p2.metric("Total XP", p.xp)
    col_p3.metric("🔥 Streak", f"{p.streak} days")
    col_p4.metric("Longest Streak", f"{p.longest_streak} days")

    # XP progress bar
    pct = int(p.level_progress_pct() * 100)
    st.markdown(f"""
    <div style="margin:12px 0 4px 0;font-size:0.9em;color:#aaa">
        Level {p.level} — {p.xp} XP &nbsp;|&nbsp; {p.xp_to_next()} XP to Level {p.level+1}
    </div>
    <div style="background:#2a2a3e;border-radius:6px;height:14px">
        <div style="background:linear-gradient(90deg,#f7971e,#ffd200);width:{pct}%;height:14px;border-radius:6px;transition:width 0.3s"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Stats ──
    st.subheader("📊 Your Stats")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Laps",     p.total_laps)
    sc2.metric("Total Sessions", p.total_sessions)
    sc3.metric("PB Improvements", p.pb_beats)
    pb_str = f"{p.personal_best_ms//60000}:{(p.personal_best_ms%60000)/1000:.3f}" if p.personal_best_ms else "—"
    sc4.metric("Personal Best",  pb_str)

    st.markdown("---")

    # ── Pending notifications ──
    if st.session_state.gami_messages:
        st.subheader("🔔 Recent Unlocks")
        for msg in st.session_state.gami_messages:
            st.success(msg)
        if st.button("Clear notifications"):
            st.session_state.gami_messages = []
            st.rerun()
        st.markdown("---")

    # ── Challenge categories ──
    challenge_map = {c.id: c for c in STATIC_CHALLENGES}
    profile_map   = {c["id"]: c for c in p.challenges}

    categories = ["milestone", "performance", "consistency", "exploration"]
    cat_labels  = {
        "milestone":   "🏁 Milestones",
        "performance": "⚡ Performance",
        "consistency": "⏱️ Consistency",
        "exploration": "🧭 Exploration",
    }

    for cat in categories:
        st.subheader(cat_labels[cat])
        cat_challenges = [c for c in STATIC_CHALLENGES if c.category == cat]
        completed = [c for c in cat_challenges if profile_map.get(c.id, {}).get("completed")]
        pending   = [c for c in cat_challenges if not profile_map.get(c.id, {}).get("completed")]

        for c in pending:
            pc = profile_map.get(c.id, {"progress": 0.0, "completed": False})
            prog = pc.get("progress", 0.0)
            tgt  = c.target if c.target > 0 else 1
            pct_c = min(int((prog / tgt) * 100), 100)
            bar_color = "#00aaff"
            st.markdown(f"""
            <div style="background:#1a1a2e;border-radius:8px;padding:12px 14px;margin:6px 0;border-left:3px solid #333">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:1.05em">{c.icon} <b>{c.name}</b></span>
                    <span style="color:#ffd200;font-size:0.9em">+{c.xp_reward} XP</span>
                </div>
                <div style="color:#aaa;font-size:0.85em;margin:4px 0">{c.description}</div>
                <div style="background:#2a2a3e;border-radius:4px;height:6px;margin-top:6px">
                    <div style="background:{bar_color};width:{pct_c}%;height:6px;border-radius:4px"></div>
                </div>
                <div style="font-size:0.75em;color:#666;margin-top:2px">{int(prog)}/{int(tgt)}</div>
            </div>
            """, unsafe_allow_html=True)

        for c in completed:
            st.markdown(f"""
            <div style="background:#0d1f0d;border-radius:8px;padding:10px 14px;margin:4px 0;border-left:3px solid #00cc66;opacity:0.75">
                <span style="color:#00cc66">✓</span> <b>{c.name}</b>
                <span style="color:#555;font-size:0.85em;margin-left:8px">{c.description}</span>
                <span style="float:right;color:#00cc66;font-size:0.85em">+{c.xp_reward} XP earned</span>
            </div>
            """, unsafe_allow_html=True)

    # ── XP log ──
    if p.xp_log:
        st.markdown("---")
        with st.expander("📜 XP History"):
            for entry in reversed(p.xp_log[-20:]):
                st.markdown(f"**+{entry['amount']} XP** — {entry['reason']} _{entry['date']}_")
