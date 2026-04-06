"""Object presence in segmentation masks query."""

import streamlit as st

from dash_query import Query
from dash_types import QueryOutput, FrameResult
from dash_loaders import load_json


class ObjectPresenceQuery(Query):
    name = "Object in Masks"
    description = "Frames containing a specific object label in segmentation masks"

    def build_params(self):
        return {
            "label": st.sidebar.text_input("Object label", "crosswalk", key="obj_l"),
            "min_score": st.sidebar.slider(
                "Min mask score", 0.0, 1.0, 0.3, 0.05, key="obj_s"
            ),
        }

    def execute(self, root, segments, params):
        target = params["label"].lower().strip()
        if not target:
            return QueryOutput([], "mask", "Enter a label", "")
        out = []
        for seg in segments:
            md = root / "annotations" / seg / "masks"
            if not md.exists():
                continue
            for mj in sorted(md.glob("*.json")):
                data = load_json(str(mj))
                hits = [
                    {**v, "mask_id": k}
                    for k, v in data.get("masks", {}).items()
                    if target in v["label"].lower()
                    and v["score"] >= params["min_score"]
                ]
                if hits:
                    out.append(FrameResult(
                        seg, mj.stem,
                        max(h["score"] for h in hits),
                        {"masks": hits},
                    ))
        out.sort(key=lambda x: -x.score)
        return QueryOutput(
            out, "mask",
            f'Frames containing "{target}"', f"{len(out)} frames",
        )
