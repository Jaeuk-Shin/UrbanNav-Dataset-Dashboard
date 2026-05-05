"""Plain image-grid visualizer with optional caption/tag text."""

from pathlib import Path

import streamlit as st

from ..clip_playback import play_button, show_selected_clip
from ..types import QueryOutput
from ._common import load_rgb


def vis_image_grid(output: QueryOutput, root: Path, max_n: int):
    show_selected_clip(root)
    items = output.results[:max_n]
    if not items:
        st.info("No results.")
        return
    cols = st.columns(4)
    for i, r in enumerate(items):
        with cols[i % 4]:
            img = load_rgb(root, r.segment, r.frame_id)
            if img is not None:
                st.image(img, caption=f"{r.segment} / {r.frame_id}",
                         use_container_width=True)
            else:
                st.warning(f"Missing: {r.segment}/{r.frame_id}")
            if r.metadata.get("caption"):
                st.caption(r.metadata["caption"][:180])
            if r.metadata.get("tags"):
                st.caption("Tags: " + ", ".join(r.metadata["tags"]))
            play_button(i, r.segment, r.frame_id, "grid")
