"""Caption / tag keyword search query."""

import streamlit as st

from ..query import Query
from ..types import QueryOutput, FrameResult
from ..loaders import load_json


class CaptionSearchQuery(Query):
    name = "Caption Search"
    description = "Frames whose caption or tags match a keyword"

    def build_params(self):
        return {
            "keyword": st.sidebar.text_input("Keyword", "tree", key="kw"),
            "in_cap": st.sidebar.checkbox("Search captions", True, key="kw_cap"),
            "in_tag": st.sidebar.checkbox("Search tags", True, key="kw_tag"),
        }

    def execute(self, root, segments, params):
        kw = params["keyword"].lower().strip()
        if not kw:
            return QueryOutput([], "image_grid", "Enter a keyword", "")
        out = []
        for seg in segments:
            f = root / "annotations" / seg / "captions.json"
            if not f.exists():
                continue
            for fid, c in load_json(str(f)).items():
                hit = (
                    (params["in_cap"] and kw in c.get("caption", "").lower())
                    or (params["in_tag"]
                        and any(kw in t.lower() for t in c.get("tags", [])))
                )
                if hit:
                    out.append(FrameResult(seg, fid, 1.0, {
                        "caption": c.get("caption", ""),
                        "tags": c.get("tags", []),
                    }))
        return QueryOutput(
            out, "image_grid",
            f'Captions matching "{kw}"', f"{len(out)} frames",
        )
