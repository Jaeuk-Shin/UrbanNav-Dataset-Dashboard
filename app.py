"""Streamlit main entry point."""

from fnmatch import fnmatch
from pathlib import Path

import streamlit as st

from dash_loaders import list_segments, load_segment_cache
from dash_queries import QUERIES
from dash_visualizers import VISUALIZERS
from dash_visualizers.table import vis_table


def main(data_root: str | None = None):
    st.set_page_config(page_title="UrbanNav Dataset Explorer", layout="wide")
    st.title("UrbanNav Dataset Explorer")

    # ── Sidebar: data root + segment filter ──────────────────────────────────
    default_root = data_root or "/home3/rvl/dataset/youtube_videos"
    root = Path(st.sidebar.text_input("Dataset root", default_root))
    has_cache = load_segment_cache(str(root)) is not None
    if not has_cache and not (root / "rgb").is_dir():
        st.error(f"`{root}/rgb` not found. Check the dataset root path.")
        return

    segs = list_segments(str(root))
    pat = st.sidebar.text_input("Segment filter (glob)", "*")
    filtered = [s for s in segs if fnmatch(s, pat)]
    st.sidebar.caption(f"{len(filtered)} / {len(segs)} segments")

    # ── Sidebar: curation DB ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    db_path = st.sidebar.text_input("Curation DB (optional)", "youtube.db",
                                    key="_db_path")
    if db_path and not Path(db_path).exists():
        st.sidebar.caption("DB not found — curation queries will be unavailable")
    # Make the DB path available to dash_clip (for trajectory rendering)
    # without threading it through every visualizer signature.
    st.session_state["_clip_db_path"] = db_path

    # ── Sidebar: query selection + params ────────────────────────────────────
    st.sidebar.markdown("---")
    qmap = {q.name: q for q in QUERIES}
    sel = st.sidebar.selectbox("Query", list(qmap))
    query = qmap[sel]
    st.sidebar.caption(query.description)
    params = query.build_params()
    params["db_path"] = db_path
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
