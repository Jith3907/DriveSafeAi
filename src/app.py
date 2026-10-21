"""
DriveSafe AI - multi-page Streamlit application.

Launch with:

    streamlit run src/app.py

Every page reuses the existing detector modules in ``src/`` unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import config
from src.ui import camera, page_live, page_placeholder, state, theme

PAGE_LABELS = {
    "live": "🟢  Live Monitor",
    "registration": "👤  Driver Registration",
    "calibration": "🎯  Calibration",
    "settings": "⚙️  Settings",
    "checklist": "✅  Test Checklist",
    "reports": "📊  Reports",
}


def _registered_driver_count() -> int:
    folder = config.KNOWN_DRIVERS_DIR
    if not folder.exists():
        return 0
    suffixes = {".npy", ".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in suffixes)


def _sidebar() -> str:
    with st.sidebar:
        st.markdown("<div class='ds-brand'>🛡️ DriveSafe AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='ds-brand-sub'>Driver monitoring suite</div>", unsafe_allow_html=True)
        st.markdown("<div class='ds-nav-title'>Navigation</div>", unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            options=list(PAGE_LABELS.keys()),
            format_func=lambda key: PAGE_LABELS[key],
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("---")
        running = bool(st.session_state.get("camera_running"))
        level = st.session_state.get("status_level", theme.LEVEL_IDLE)
        st.markdown(
            "<div class='ds-side-foot'>"
            f"<b>Camera:</b> {'live' if running else 'off'}<br>"
            f"<b>Status:</b> {level}<br>"
            f"<b>Drivers registered:</b> {_registered_driver_count()}<br>"
            f"<b>Model:</b> {'ready' if camera.model_available() else 'missing'}"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("Run with `streamlit run src/app.py`")
        st.caption("Logs: `logs/events.csv` · Calibration: `data/calibration.json`")

    return page


def _handle_page_change(page: str) -> None:
    """Stops the camera loop and releases the webcam when leaving the Live Monitor."""
    previous = st.session_state.get("last_page")
    st.session_state.last_page = page

    changed = previous is not None and previous != page
    if (changed or page != "live") and st.session_state.get("camera_running"):
        state.close_active_events()
        camera.stop_camera()
        # The sidebar was already painted with the old camera state; repaint it.
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="DriveSafe AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    state.init_state()
    state.ensure_events_csv()
    theme.inject_css()
    camera.cleanup_stale_worker()

    page = _sidebar()
    _handle_page_change(page)

    if page == "live":
        page_live.render()
    else:
        theme.page_header(theme.LEVEL_IDLE)
        page_placeholder.render(page)


# Streamlit executes this script as __main__. The guard keeps `multiprocessing`
# (spawn) from re-running the UI when the detector process imports this module.
if __name__ == "__main__":
    main()
