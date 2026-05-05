"""Detection bounding-box overlay visualizer.

Accepts either ``pedestrians`` or ``crosswalks`` in ``metadata`` (or both),
drawing each class with a distinct colour.
"""

from pathlib import Path

import cv2
import streamlit as st

from dash_clip import play_button, show_selected_clip
from dash_types import QueryOutput
from dash_visualizers._common import PALETTE, load_rgb


def _draw(img, boxes, color):
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'{b["confidence"]:.2f}', (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def vis_detection(output: QueryOutput, root: Path, max_n: int):
    show_selected_clip(root)
    items = output.results[:max_n]
    if not items:
        st.info("No results.")
        return
    cols = st.columns(4)
    for i, r in enumerate(items):
        with cols[i % 4]:
            img = load_rgb(root, r.segment, r.frame_id)
            if img is None:
                st.warning("Missing frame")
                continue

            peds = r.metadata.get("pedestrians", [])
            xws = r.metadata.get("crosswalks", [])

            _draw(img, peds, PALETTE[0])        # red
            _draw(img, xws, PALETTE[1])         # green

            parts = []
            if peds:
                parts.append(f"{len(peds)} peds")
            if xws:
                parts.append(f"{len(xws)} xw")
            tag = f" [{', '.join(parts)}]" if parts else ""

            st.image(
                img,
                caption=f"{r.segment}/{r.frame_id}{tag}",
                use_container_width=True,
            )
            play_button(i, r.segment, r.frame_id, "det")
