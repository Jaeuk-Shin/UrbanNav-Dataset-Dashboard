"""Pedestrian count query."""

import streamlit as st

from query import Query
from types import QueryOutput, FrameResult
from loaders import load_json


class PedestrianCountQuery(Query):
    name = "Pedestrian Count"
    description = "Frames with N+ pedestrians above a confidence threshold"

    def build_params(self):
        return {
            "min_count": st.sidebar.slider("Min pedestrians", 1, 20, 3, key="ped_n"),
            "min_conf": st.sidebar.slider(
                "Min confidence", 0.0, 1.0, 0.5, 0.05, key="ped_c"
            ),
        }

    def execute(self, root, segments, params):
        out = []
        for seg in segments:
            f = root / "annotations" / seg / "detections.json"
            if not f.exists():
                continue
            for fid, d in load_json(str(f)).items():
                good = [
                    p for p in d["pedestrians"]
                    if p["confidence"] >= params["min_conf"]
                ]
                if len(good) >= params["min_count"]:
                    out.append(FrameResult(seg, fid, len(good),
                                           {"pedestrians": good}))
        out.sort(key=lambda x: -x.score)
        return QueryOutput(
            out, "detection",
            f"Frames with >= {params['min_count']} pedestrians "
            f"(conf >= {params['min_conf']})",
            f"{len(out)} frames",
        )
