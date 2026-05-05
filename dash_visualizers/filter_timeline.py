"""Per-segment filter-metric timeline visualizer.

Shows 8 stacked subplots (one per metric) with threshold lines and
green/red background indicating per-filter pass/fail regions.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dash_types import QueryOutput

# (subplot_title, metric_key, mask_key, threshold_key)
_PANELS = [
    ("Velocity", "velocities", "velocity_spike", "velocity_spike"),
    ("Forward-Camera Angle", "forward_camera_angles",
     "forward_camera_angle", "forward_camera_angle"),
    ("Roll Change", "roll_changes", "roll_change", "roll_change"),
    ("Pitch Change", "pitch_changes", "pitch_change", "pitch_change"),
    ("Yaw Change", "yaw_changes", "yaw_change", "yaw_change"),
    ("Abs Pitch", "abs_pitch", "abs_pitch", "abs_pitch"),
    ("Abs Roll", "abs_roll", "abs_roll", "abs_roll"),
    ("Height Change", "height_changes", "height_change", "height_change"),
]


def _mask_to_regions(mask: list[bool]) -> list[tuple[int, int, bool]]:
    """Convert a boolean mask to contiguous (start, end, value) regions."""
    if not mask:
        return []
    regions = []
    start = 0
    val = mask[0]
    for i in range(1, len(mask)):
        if mask[i] != val:
            regions.append((start, i - 1, val))
            start = i
            val = mask[i]
    regions.append((start, len(mask) - 1, val))
    return regions


def vis_filter_timeline(output: QueryOutput, root: Path, max_n: int):
    items = output.results[:max_n]
    if not items:
        st.info("No results.")
        return

    for r in items:
        md = r.metadata
        seg = md["segment"]
        n = md["num_frames"]
        pct = md["pass_rate"]
        metrics = md["metrics"]
        masks = md["masks"]
        thresholds = md["thresholds"]

        st.markdown(f"### {seg}  &mdash;  {n} frames, {pct:.1f}% valid")

        # Build combined mask strip
        combined = masks.get("combined", [True] * n)

        n_panels = len(_PANELS) + 1  # +1 for combined mask
        fig = make_subplots(
            rows=n_panels, cols=1, shared_xaxes=True,
            vertical_spacing=0.015,
            subplot_titles=[p[0] for p in _PANELS] + ["Combined"],
            row_heights=[1] * len(_PANELS) + [0.3],
        )

        x = list(range(n))

        for idx, (title, metric_key, mask_key, thresh_key) in enumerate(_PANELS):
            row = idx + 1
            vals = metrics.get(metric_key, [])
            mask = masks.get(mask_key)

            # Background regions (green/red) from individual filter mask
            if mask is not None:
                for s, e, v in _mask_to_regions(mask):
                    fig.add_vrect(
                        x0=s - 0.5, x1=e + 0.5,
                        fillcolor="rgba(0,180,0,0.08)" if v
                        else "rgba(220,0,0,0.12)",
                        line_width=0, row=row, col=1,
                    )

            # Metric line
            if vals:
                fig.add_trace(go.Scatter(
                    x=x[:len(vals)], y=vals,
                    mode="lines", line=dict(color="#1f77b4", width=1),
                    showlegend=False,
                ), row=row, col=1)

            # Threshold line
            thresh = thresholds.get(thresh_key)
            if thresh is not None:
                fig.add_hline(
                    y=thresh, line_dash="dash",
                    line_color="red", line_width=1,
                    annotation_text=f"{thresh}",
                    annotation_position="top right",
                    row=row, col=1,
                )

        # Combined mask as coloured strip
        combined_row = n_panels
        for s, e, v in _mask_to_regions(combined):
            fig.add_vrect(
                x0=s - 0.5, x1=e + 0.5,
                fillcolor="rgba(0,180,0,0.3)" if v
                else "rgba(220,0,0,0.3)",
                line_width=0, row=combined_row, col=1,
            )
        # Invisible trace so the row renders
        fig.add_trace(go.Scatter(
            x=[0, n - 1], y=[0, 0], mode="lines",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ), row=combined_row, col=1)

        fig.update_layout(
            height=100 * n_panels,
            xaxis=dict(title="Frame Index"),
            margin=dict(l=60, r=20, t=30, b=40),
            showlegend=False,
        )
        # Hide y-axis for the combined strip
        fig.update_yaxes(visible=False, row=combined_row, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Summary table: which filters reject how many frames
        st.markdown("**Per-filter rejection:**")
        reject_data = []
        for _, _, mask_key, _ in _PANELS:
            mask = masks.get(mask_key)
            if mask is not None:
                rejected = sum(1 for v in mask if not v)
                reject_data.append({
                    "filter": mask_key,
                    "rejected_frames": rejected,
                    "pct": f"{100.0 * rejected / n:.1f}%"
                         if n > 0 else "0%",
                })
        # Add sustained_slow and stop_without_reasons
        for extra_key in ("sustained_slow", "stop_without_reasons"):
            mask = masks.get(extra_key)
            if mask is not None:
                rejected = sum(1 for v in mask if not v)
                reject_data.append({
                    "filter": extra_key,
                    "rejected_frames": rejected,
                    "pct": f"{100.0 * rejected / n:.1f}%"
                         if n > 0 else "0%",
                })
        if reject_data:
            import pandas as pd
            st.dataframe(pd.DataFrame(reject_data), use_container_width=True)
