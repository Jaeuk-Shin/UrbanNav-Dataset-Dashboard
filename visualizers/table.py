"""Sortable dataframe + CSV-download visualizer."""

from pathlib import Path

import pandas as pd
import streamlit as st

from ..types import QueryOutput


def vis_table(output: QueryOutput, root: Path, max_n: int):
    items = output.results[:max_n]
    if not items:
        st.info("No data.")
        return
    df = pd.DataFrame([r.metadata for r in items])
    st.dataframe(df, use_container_width=True, height=600)
    st.download_button("Download CSV", df.to_csv(index=False),
                       "query_results.csv", "text/csv")
