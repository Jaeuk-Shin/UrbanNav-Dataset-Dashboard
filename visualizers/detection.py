"""Detection bounding-box overlay visualizer."""

from pathlib import Path

import cv2
import streamlit as st

from types import QueryOutput
from visualizers._common import PALETTE, load_rgb


def vis_detection(output: QueryOutput, root: Path, max_n: int):
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
            color = PALETTE[0]
            for ped in r.metadata.get("pedestrians", []):
                x1, y1, x2, y2 = (int(v) for v in ped["bbox"])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f'{ped["confidence"]:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            st.image(
                img,
                caption=f"{r.segment}/{r.frame_id} [{int(r.score)} peds]",
                use_container_width=True,
            )
