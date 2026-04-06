"""XZ trajectory + velocity profile visualizer."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from types import QueryOutput


def vis_trajectory(output: QueryOutput, root: Path, max_n: int):
    items = output.results[:max_n]
    if not items:
        st.info("No results.")
        return
    for r in items:
        pos = np.array(r.metadata["positions"])   # (N, 3) = x y z
        vel = np.array(r.metadata["velocities"])   # (N-1,)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(
                x=pos[:-1, 0], y=pos[:-1, 2],
                mode="markers+lines",
                marker=dict(size=4, color=vel, colorscale="Turbo",
                            showscale=True, colorbar=dict(title="m/s")),
                line=dict(color="rgba(180,180,180,0.4)", width=1),
            ))
            fig.update_layout(
                title=r.segment,
                xaxis_title="X (m)", yaxis_title="Z (m)",
                height=370, yaxis=dict(scaleanchor="x"),
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(go.Scatter(
                y=vel, mode="lines", line=dict(width=1),
            ))
            fig.update_layout(
                title=(f"avg {r.metadata['avg_velocity']} / "
                       f"max {r.metadata['max_velocity']} m/s"),
                xaxis_title="Frame", yaxis_title="Velocity (m/s)",
                height=370,
            )
            st.plotly_chart(fig, use_container_width=True)
