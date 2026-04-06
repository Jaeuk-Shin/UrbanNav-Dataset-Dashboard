"""Streamlit main entry point."""

from fnmatch import fnmatch
from pathlib import Path

import streamlit as st

from loaders import list_segments
from queries import QUERIES
from visualizers import VISUALIZERS
from visualizers.table import vis_table


def main(data_root: str | None = None):
    st.set_page_config(page_title="UrbanNav Dataset Explorer", layout="wide")
    st.title("UrbanNav Dataset Explorer")

    # ── Sidebar: data root + segment filter ──────────────────────────────────
    default_root = data_root or "/raid/robot/real_world_dataset/omr/dataset"
    root = Path(st.sidebar.text_input("Dataset root", default_root))
    if not (root / "rgb").is_dir():
        st.error(f"`{root}/rgb` not found. Check the dataset root path.")
        return

    segs = list_segments(str(root / "rgb"))
    pat = st.sidebar.text_input("Segment filter (glob)", "*")
    filtered = [s for s in segs if fnmatch(s, pat)]
    st.sidebar.caption(f"{len(filtered)} / {len(segs)} segments")

    # ── Sidebar: query selection + params ────────────────────────────────────
    st.sidebar.markdown("---")
    qmap = {q.name: q for q in QUERIES}
    sel = st.sidebar.selectbox("Query", list(qmap))
    query = qmap[sel]
    st.sidebar.caption(query.description)
    params = query.build_params()
    max_n = st.sidebar.slider("Max results", 10, 200, 40, key="_max_n")

    # Clear stale results when switching query type
    if st.session_state.get("_last_q") != sel:
        st.session_state.pop("_output", None)
        st.session_state["_last_q"] = sel

    # ── Execute ──────────────────────────────────────────────────────────────
    if st.sidebar.button("Run Query", type="primary"):
        with st.spinner("Querying ..."):
            st.session_state["_output"] = query.execute(root, filtered, params)

    # ── Render results ───────────────────────────────────────────────────────
    out = st.session_state.get("_output")
    if out is not None:
        st.subheader(out.title)
        st.caption(out.description)
        VISUALIZERS.get(out.viz_type, vis_table)(out, root, max_n)
    else:
        st.info("Select a query and click **Run Query** to explore the dataset.")
