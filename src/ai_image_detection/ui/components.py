"""Reusable Streamlit UI components."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go


_VERDICT_CONFIG = {
    "Authentic":        {"color": "#22c55e", "bg": "#f0fdf4", "icon": "✅"},
    "Suspicious":       {"color": "#f59e0b", "bg": "#fffbeb", "icon": "⚠️"},
    "Likely Fraudulent":{"color": "#ef4444", "bg": "#fef2f2", "icon": "🚨"},
}

_CONFIDENCE_CONFIG = {
    "HIGH":   {"color": "#22c55e", "label": "HIGH"},
    "MEDIUM": {"color": "#f59e0b", "label": "MEDIUM"},
    "LOW":    {"color": "#ef4444", "label": "LOW"},
}


def verdict_banner(verdict: str, score: float, confidence: str) -> None:
    cfg = _VERDICT_CONFIG.get(verdict, _VERDICT_CONFIG["Suspicious"])
    conf_cfg = _CONFIDENCE_CONFIG.get(confidence, _CONFIDENCE_CONFIG["LOW"])
    st.markdown(
        f"""
        <div style="
            background:{cfg['bg']};
            border-left: 6px solid {cfg['color']};
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 16px;
        ">
            <span style="font-size:2rem;">{cfg['icon']}</span>
            <span style="font-size:1.6rem; font-weight:700; color:{cfg['color']}; margin-left:12px;">
                {verdict}
            </span>
            <div style="margin-top:8px; color:#374151;">
                Ensemble score: <b>{score:.4f}</b> &nbsp;|&nbsp;
                Confidence: <b style="color:{conf_cfg['color']}">{conf_cfg['label']}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_bar_chart(signal_breakdown: list) -> None:
    available = [r for r in signal_breakdown if r.get("available") and r.get("score") is not None]
    if not available:
        st.info("No signals available to chart.")
        return

    labels  = [r["label"] for r in available]
    scores  = [r["score"] for r in available]
    weights = [r.get("weight_used", r.get("base_weight", 0.0)) for r in available]

    colors = []
    for s in scores:
        if s < 0.35:
            colors.append("#22c55e")
        elif s < 0.60:
            colors.append("#f59e0b")
        else:
            colors.append("#ef4444")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.3f}  (w={w:.2f})" for s, w in zip(scores, weights)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.15], title="Fake Probability"),
        yaxis=dict(autorange="reversed"),
        height=max(250, len(available) * 48),
        margin=dict(l=10, r=60, t=10, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.add_vline(x=0.35, line_dash="dot", line_color="#22c55e", annotation_text="Auth", annotation_position="top")
    fig.add_vline(x=0.60, line_dash="dot", line_color="#ef4444", annotation_text="Fraud", annotation_position="top")
    st.plotly_chart(fig, use_container_width=True)


def signal_breakdown_table(signal_breakdown: list) -> None:
    rows = []
    for r in signal_breakdown:
        status = "✅" if r.get("available") else "⏭️ skipped"
        score_str = f"{r['score']:.4f}" if r.get("score") is not None else "—"
        weight_str = f"{r.get('weight_used', r.get('base_weight', 0)):.2f}"
        rows.append({
            "Signal":   r["label"],
            "Score":    score_str,
            "Weight":   weight_str,
            "Status":   status,
        })
    st.table(rows)


def skipped_signals_warning(skipped: list) -> None:
    if not skipped:
        return
    with st.expander(f"⏭️ {len(skipped)} signal(s) skipped", expanded=False):
        for s in skipped:
            st.markdown(f"- `{s}`")
        st.caption("Skipped signals were excluded from scoring. Remaining weights were renormalised.")
