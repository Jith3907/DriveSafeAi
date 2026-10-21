"""
Shared light-theme styling and reusable HTML components for the DriveSafe AI app.

Everything visual lives here so each page stays focused on behaviour.
"""

from __future__ import annotations

import base64
import html

import streamlit as st

# ---------------------------------------------------------
# Status levels used by the banner, header chip and messages
# ---------------------------------------------------------
LEVEL_SAFE = "SAFE"
LEVEL_ATTENTION = "ATTENTION"
LEVEL_DANGER = "DANGER"
LEVEL_IDLE = "IDLE"

_CSS = """
<style>
    :root { color-scheme: light; }

    .stApp { background: #f5f8fc; }
    .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1500px; }
    h1, h2, h3, h4, h5 { color: #0f172a; letter-spacing: -0.01em; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
        padding-top: 0.6rem;
    }
    .ds-brand {
        display: flex; align-items: center; gap: 10px;
        font-size: 1.15rem; font-weight: 800; color: #0f172a;
        padding: 4px 2px 2px 2px;
    }
    .ds-brand-sub { font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.08em;
        text-transform: uppercase; margin: 0 0 10px 2px; }
    .ds-nav-title { font-size: 0.7rem; font-weight: 800; letter-spacing: 0.1em;
        text-transform: uppercase; color: #94a3b8; margin: 6px 0 2px 2px; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 7px 8px; border-radius: 8px; transition: background 0.15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #f1f5f9; }
    .ds-side-foot { font-size: 0.75rem; color: #64748b; line-height: 1.5; }

    /* Header */
    .ds-header {
        display: flex; align-items: center; justify-content: space-between; gap: 18px;
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 14px 22px; margin-bottom: 14px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }
    .ds-header-title { font-size: 1.55rem; font-weight: 800; color: #0f172a; line-height: 1.2; }
    .ds-header-sub { font-size: 0.82rem; color: #64748b; margin-top: 3px; }

    /* Status chip */
    .ds-chip {
        display: inline-flex; align-items: center; gap: 8px;
        font-weight: 800; font-size: 0.9rem; letter-spacing: 0.08em;
        padding: 8px 16px; border-radius: 999px; border: 1px solid; white-space: nowrap;
    }
    .ds-chip .ds-dot { width: 9px; height: 9px; border-radius: 50%;
        background: currentColor; box-shadow: 0 0 0 3px rgba(15, 23, 42, 0.06); }
    .ds-chip-safe { background: #e8f8ee; color: #15803d; border-color: #86efac; }
    .ds-chip-attention { background: #fef3c7; color: #b45309; border-color: #fcd34d; }
    .ds-chip-danger { background: #fee2e2; color: #b91c1c; border-color: #fca5a5; }
    .ds-chip-idle { background: #eef2f7; color: #64748b; border-color: #cbd5e1; }
    .ds-chip-live .ds-dot { animation: ds-pulse 1.4s ease-in-out infinite; }
    @keyframes ds-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    /* Big status banner */
    .ds-banner {
        border-radius: 14px; padding: 16px 22px; color: #ffffff;
        font-weight: 800; font-size: 1.4rem; text-align: center;
        margin-bottom: 12px; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
    }
    .ds-banner-sub { display: block; font-size: 0.74rem; font-weight: 700;
        opacity: 0.92; margin-top: 3px; letter-spacing: 0.08em; text-transform: uppercase; }
    .ds-banner-safe { background: linear-gradient(135deg, #16a34a 0%, #15803d 100%); }
    .ds-banner-attention { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    .ds-banner-danger { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%); }
    .ds-banner-idle { background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%); }

    /* Indicator cards */
    .ds-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .ds-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid #cbd5e1;
        border-radius: 10px; padding: 9px 12px;
    }
    .ds-card-label { font-size: 0.68rem; font-weight: 800; letter-spacing: 0.08em;
        text-transform: uppercase; color: #64748b; }
    .ds-card-value { font-size: 0.98rem; font-weight: 700; color: #334155; margin-top: 2px; }
    .ds-card-detail { font-size: 0.72rem; color: #94a3b8; margin-top: 1px; }
    .ds-card.on-red { border-left-color: #dc2626; background: #fef2f2; }
    .ds-card.on-red .ds-card-value { color: #b91c1c; }
    .ds-card.on-amber { border-left-color: #d97706; background: #fffbeb; }
    .ds-card.on-amber .ds-card-value { color: #b45309; }
    .ds-card.on-green { border-left-color: #16a34a; background: #f0fdf4; }
    .ds-card.on-green .ds-card-value { color: #15803d; }

    /* Metric tiles */
    .ds-metrics { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 4px 0 0 0; }
    .ds-metric { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 9px;
        padding: 8px 4px; text-align: center; }
    .ds-metric-label { font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.08em;
        color: #94a3b8; font-weight: 800; }
    .ds-metric-value { font-size: 1.12rem; font-weight: 800; color: #0f172a;
        font-variant-numeric: tabular-nums; }
    .ds-metric-value.good { color: #15803d; }
    .ds-metric-value.bad { color: #b91c1c; }

    /* Messages box */
    .ds-messages { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
        max-height: 208px; overflow-y: auto; padding: 6px 12px; }
    .ds-msg { display: flex; gap: 8px; align-items: flex-start;
        padding: 6px 0; border-bottom: 1px dashed #eef2f7; font-size: 0.82rem; color: #334155; }
    .ds-msg:last-child { border-bottom: none; }
    .ds-msg-time { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        color: #94a3b8; font-size: 0.75rem; white-space: nowrap; padding-top: 1px; }
    .ds-level { font-size: 0.62rem; font-weight: 800; letter-spacing: 0.06em;
        padding: 2px 6px; border-radius: 5px; white-space: nowrap; }
    .ds-level-danger { background: #fee2e2; color: #b91c1c; }
    .ds-level-attention { background: #fef3c7; color: #b45309; }
    .ds-level-safe { background: #dcfce7; color: #15803d; }
    .ds-level-info { background: #e2e8f0; color: #475569; }
    .ds-empty { color: #94a3b8; font-size: 0.82rem; padding: 12px 4px; }

    /* Section titles + debug table */
    .ds-section-title { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.09em;
        text-transform: uppercase; color: #64748b; margin: 12px 0 6px 0; }
    .ds-debug { width: 100%; border-collapse: collapse; font-size: 0.8rem;
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }
    .ds-debug th { text-align: left; color: #64748b; font-weight: 800; font-size: 0.66rem;
        text-transform: uppercase; letter-spacing: 0.07em;
        border-bottom: 1px solid #e2e8f0; padding: 7px 10px; background: #f8fafc; }
    .ds-debug td { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; color: #334155; }
    .ds-debug tr:last-child td { border-bottom: none; }
    .ds-debug td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .ds-note { font-size: 0.85rem; color: #64748b; }
    .ds-reason { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 9px;
        padding: 9px 12px; font-size: 0.83rem; color: #334155; margin-top: 8px; }
    .ds-placeholder { background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 14px;
        padding: 28px; text-align: center; color: #64748b; }

    /* Live video */
    .ds-video { width: 100%; display: block; border-radius: 12px;
        border: 1px solid #e2e8f0; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08); }
</style>
"""


def inject_css() -> None:
    """Injects the shared stylesheet once per page render."""
    st.markdown(_CSS, unsafe_allow_html=True)


def status_chip_html(level: str, live: bool = False) -> str:
    """Returns the markup for the small SAFE / ATTENTION / DANGER pill."""
    level = (level or LEVEL_IDLE).upper()
    css = {
        LEVEL_SAFE: "ds-chip-safe",
        LEVEL_ATTENTION: "ds-chip-attention",
        LEVEL_DANGER: "ds-chip-danger",
    }.get(level, "ds-chip-idle")
    label = level if level in (LEVEL_SAFE, LEVEL_ATTENTION, LEVEL_DANGER) else "OFFLINE"
    live_css = " ds-chip-live" if live else ""
    return (
        f"<span class='ds-chip {css}{live_css}'>"
        f"<span class='ds-dot'></span>{html.escape(label)}</span>"
    )


def page_header(level: str = LEVEL_IDLE, subtitle: str | None = None, live: bool = False) -> None:
    """Renders the app header with the project name and the live status chip."""
    subtitle = subtitle or (
        "Real-time driver monitoring &middot; drowsiness &middot; distraction "
        "&middot; phone usage &middot; identity"
    )
    st.markdown(
        f"""
        <div class="ds-header">
            <div>
                <div class="ds-header-title">&#128737;&#65039; DriveSafe AI</div>
                <div class="ds-header-sub">{subtitle}</div>
            </div>
            {status_chip_html(level, live=live)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def banner_html(level: str, headline: str | None = None, sub: str | None = None) -> str:
    """Returns the markup for the large status banner."""
    level = (level or LEVEL_IDLE).upper()
    css = {
        LEVEL_SAFE: "ds-banner-safe",
        LEVEL_ATTENTION: "ds-banner-attention",
        LEVEL_DANGER: "ds-banner-danger",
    }.get(level, "ds-banner-idle")
    default_headline = {
        LEVEL_SAFE: "SAFE",
        LEVEL_ATTENTION: "ATTENTION",
        LEVEL_DANGER: "DANGER",
    }.get(level, "MONITORING PAUSED")
    default_sub = {
        LEVEL_SAFE: "Driver attentive &amp; road focused",
        LEVEL_ATTENTION: "Driver distraction detected",
        LEVEL_DANGER: "Immediate driver attention required",
    }.get(level, "Start the camera to begin monitoring")
    headline = headline or default_headline
    sub = sub or default_sub
    return (
        f"<div class='ds-banner {css}'>{html.escape(headline)}"
        f"<span class='ds-banner-sub'>{sub}</span></div>"
    )


def video_html(jpeg_bytes: bytes) -> str:
    """Renders a JPEG frame as an inline data URI.

    ``st.image`` inside a fragment that reruns ~10x/second points at media URLs
    that Streamlit deletes between runs (the browser then shows a broken image),
    so the live feed is embedded directly instead.
    """
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return (
        "<img class='ds-video' alt='Live camera feed' "
        f"src='data:image/jpeg;base64,{encoded}' />"
    )


def indicator_grid_html(cards: list[dict]) -> str:
    """
    Returns the 2-column indicator card grid.

    Each card is ``{"label": str, "value": str, "detail": str, "state": "off|red|amber|green"}``.
    """
    tiles = []
    for card in cards:
        state = card.get("state", "off")
        state_css = state if state in ("red", "amber", "green") else ""
        tiles.append(
            "<div class='ds-card {on}'>"
            "<div class='ds-card-label'>{label}</div>"
            "<div class='ds-card-value'>{value}</div>"
            "<div class='ds-card-detail'>{detail}</div>"
            "</div>".format(
                on=f"on-{state_css}" if state_css else "",
                label=html.escape(str(card.get("label", ""))),
                value=html.escape(str(card.get("value", ""))),
                detail=html.escape(str(card.get("detail", ""))),
            )
        )
    return "<div class='ds-grid'>" + "".join(tiles) + "</div>"


def metrics_html(metrics: list[dict]) -> str:
    """Returns the row of live metric tiles (EAR, MAR, yaw, pitch, FPS...)."""
    tiles = []
    for metric in metrics:
        value_class = metric.get("tone", "")
        tiles.append(
            "<div class='ds-metric'>"
            "<div class='ds-metric-label'>{label}</div>"
            "<div class='ds-metric-value {tone}'>{value}</div>"
            "</div>".format(
                label=html.escape(str(metric.get("label", ""))),
                tone=value_class if value_class in ("good", "bad") else "",
                value=html.escape(str(metric.get("value", "-"))),
            )
        )
    return "<div class='ds-metrics'>" + "".join(tiles) + "</div>"


def messages_html(messages: list[dict]) -> str:
    """
    Returns the Messages box markup.

    Each message is ``{"ts": "12:04:31", "level": "DANGER", "text": "Phone usage detected"}``,
    newest first.
    """
    if not messages:
        return (
            "<div class='ds-messages'><div class='ds-empty'>"
            "No alerts yet &mdash; driver attentive.</div></div>"
        )
    rows = []
    for msg in messages:
        level = str(msg.get("level", "INFO")).upper()
        level_css = {
            LEVEL_DANGER: "ds-level-danger",
            LEVEL_ATTENTION: "ds-level-attention",
            LEVEL_SAFE: "ds-level-safe",
        }.get(level, "ds-level-info")
        rows.append(
            "<div class='ds-msg'>"
            "<span class='ds-msg-time'>{ts}</span>"
            "<span class='ds-level {css}'>{level}</span>"
            "<span>{text}</span>"
            "</div>".format(
                ts=html.escape(str(msg.get("ts", ""))),
                css=level_css,
                level=html.escape(level),
                text=html.escape(str(msg.get("text", ""))),
            )
        )
    return "<div class='ds-messages'>" + "".join(rows) + "</div>"


def debug_table_html(rows: list[tuple[str, str, str]]) -> str:
    """Returns a 3-column (signal / value / flag) debug telemetry table."""
    body = "".join(
        "<tr><td>{name}</td><td class='num'>{value}</td><td class='num'>{flag}</td></tr>".format(
            name=html.escape(str(name)),
            value=html.escape(str(value)),
            flag=html.escape(str(flag)),
        )
        for name, value, flag in rows
    )
    return (
        "<table class='ds-debug'><thead><tr>"
        "<th>Signal</th><th style='text-align:right'>Value</th>"
        "<th style='text-align:right'>State</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def placeholder_html(title: str, description: str) -> str:
    return (
        "<div class='ds-placeholder'>"
        f"<div style='font-size:1.1rem; font-weight:700; color:#334155;'>{html.escape(title)}</div>"
        f"<div style='margin-top:6px;'>{description}</div>"
        "</div>"
    )
