"""Placeholder pages for the sections that follow the Live Monitor build step."""

from __future__ import annotations

import streamlit as st

from src.ui import theme

PAGE_SPECS: dict[str, dict] = {
    "registration": {
        "title": "👤 Driver Registration",
        "summary": "Register the person behind the wheel and manage known drivers.",
        "bullets": [
            "Form with driver name (required) and optional ID / note.",
            "Capture Face: 3-second countdown, capture, save the embedding to "
            "`data/known_drivers/<name>.npy`.",
            "st.success / st.error feedback, for example "
            "“Driver Alex registered successfully” or “No face detected, try again”.",
            "Table of registered drivers with a Delete button on every row.",
        ],
    },
    "calibration": {
        "title": "🎯 Calibration",
        "summary": "Guided, step-by-step calibration for this driver and this camera position.",
        "bullets": [
            "Neutral pose (3 s), eyes open (5 s), eyes closed (5 s) and yawn captures.",
            "Progress bars and per-step instructions while capturing.",
            "Measured values next to suggested thresholds.",
            "Save writes `data/calibration.json`.",
        ],
    },
    "settings": {
        "title": "⚙️ Settings",
        "summary": "Tune every detection threshold and timer in one place.",
        "bullets": [
            "Sliders for EAR, MAR, head-turn angle, looking-away timers, face-lost timers, "
            "PHONE_CONF, vote window and hold time.",
            "Save persists to `data/calibration.json` and keeps settings across page switches.",
            "Reset to defaults with a confirmation message.",
        ],
    },
    "checklist": {
        "title": "✅ Test Checklist",
        "summary": "Manual acceptance testing with live feedback from the running detectors.",
        "bullets": [
            "One row per test: instruction, live “detected now” indicator, PASS / FAIL buttons.",
            "You mark the result yourself - no automatic PASS.",
            "Results table plus a Download CSV button.",
        ],
    },
    "reports": {
        "title": "📊 Reports",
        "summary": "Review everything that happened, from `logs/events.csv`.",
        "bullets": [
            "Filters: date range, event type and driver.",
            "Summary cards: total events, count per type, longest drowsy episode, "
            "monitoring time, percentage of time SAFE.",
            "Charts: events per type (bar) and events over time (timeline).",
            "Full event table with Download CSV and Download HTML report.",
            "Clear logs with a confirmation step.",
        ],
    },
}


def render(page_key: str) -> None:
    spec = PAGE_SPECS.get(page_key)
    if spec is None:
        st.warning("Unknown page.")
        return

    st.markdown(
        theme.placeholder_html(
            spec["title"],
            f"{spec['summary']}<br><br>"
            "<span style='font-size:0.8rem; color:#94a3b8;'>"
            "This page is built in the next step - navigation, header and session state already work."
            "</span>",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("**Planned behaviour**")
    for bullet in spec["bullets"]:
        st.markdown(f"- {bullet}")
