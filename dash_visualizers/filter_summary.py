"""Aggregate filter-statistics visualizer: metric cards, charts, tables."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dash_types import QueryOutput


def vis_filter_summary(output: QueryOutput, root: Path, max_n: int):
    if not output.results:
        st.info("No data.")
        return

    agg = output.results[0].metadata.get("_aggregate")
    if agg is None:
        st.warning("Missing aggregate data.")
        return

    # ── Row 1: metric cards ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Segments", f"{agg['total_segments']:,}")
    c2.metric("Alive", f"{agg['alive_segments']:,}")
    c3.metric("Dead", f"{agg['dead_segments']:,}")
    frame_pct = (100.0 * agg["valid_frames"] / agg["total_frames"]
                 if agg["total_frames"] else 0)
    c4.metric("Frame Pass Rate", f"{frame_pct:.1f}%")

    # ── Row 2: bar chart + histogram ────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Per-Filter Rejection")
        pf = agg.get("per_filter", {})
        if pf:
            sorted_pf = sorted(pf.items(), key=lambda x: x[1], reverse=True)
            names = [k for k, _ in sorted_pf]
            counts = [v for _, v in sorted_pf]
            fig = go.Figure(go.Bar(
                x=counts, y=names, orientation="h",
                marker_color="indianred",
            ))
            fig.update_layout(
                xaxis_title="Rejected Frames",
                yaxis=dict(autorange="reversed"),
                height=400, margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Segment Pass-Rate Distribution")
        hist_bins = agg.get("pass_rate_histogram")
        if hist_bins:
            bin_edges = [i * 5 for i in range(len(hist_bins))]
            fig = go.Figure(go.Bar(
                x=[f"{e}%" for e in bin_edges],
                y=hist_bins,
                marker_color="steelblue",
            ))
            fig.update_layout(
                xaxis_title="Pass Rate",
                yaxis_title="Segments",
                height=400, margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: per-video table ──────────────────────────────────────────
    per_video = agg.get("per_video")
    if per_video:
        st.subheader("Per-Video Breakdown")
        df = pd.DataFrame(per_video)
        df = df.sort_values("pass_rate", ascending=True)
        st.dataframe(df, use_container_width=True, height=400)
        st.download_button("Download CSV", df.to_csv(index=False),
                           "filter_per_video.csv", "text/csv",
                           key="co_csv")

    # ── Row 4: per-segment table (from results) ────────────────────────
    items = [r for r in output.results
             if "_aggregate" not in r.metadata][:max_n]
    if items:
        st.subheader("Per-Segment Detail")
        rows = [{k: v for k, v in r.metadata.items() if k != "_aggregate"}
                for r in items]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)
