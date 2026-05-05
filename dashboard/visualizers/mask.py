"""Segmentation mask overlay visualizer."""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from ..clip_playback import play_button, show_selected_clip
from ..types import QueryOutput
from ._common import PALETTE, load_rgb


def mask(output: QueryOutput, root: Path, max_n: int):
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
            # Overlay mask pixels
            mask_f = (root / "annotations" / r.segment
                      / "masks" / f"{r.frame_id}.png")
            if mask_f.exists():
                mask = cv2.imread(str(mask_f), cv2.IMREAD_UNCHANGED)  # uint16
                if mask is not None and mask.shape[:2] == img.shape[:2]:
                    overlay = np.zeros_like(img)
                    for j, m in enumerate(r.metadata.get("masks", [])):
                        overlay[mask == int(m["mask_id"])] = \
                            PALETTE[j % len(PALETTE)]
                    hit = overlay.sum(axis=2) > 0
                    img[hit] = (img[hit] * 0.6
                                + overlay[hit] * 0.4).astype(np.uint8)
            # Bounding boxes
            for j, m in enumerate(r.metadata.get("masks", [])):
                c = PALETTE[j % len(PALETTE)]
                x1, y1, x2, y2 = (int(v) for v in m["bbox"])
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(img, f'{m["label"]} {m["score"]:.2f}',
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
            labels = ", ".join(m["label"] for m in r.metadata.get("masks", []))
            st.image(img, caption=f"{r.segment}/{r.frame_id}\n{labels}",
                     use_container_width=True)
            play_button(i, r.segment, r.frame_id, "mask")
