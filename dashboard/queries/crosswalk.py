"""Crosswalk detection query, backed by youtube.db.

Crosswalks are populated by ``stages/crosswalk.py`` as ``crosswalks.json`` per
segment, then materialised into the ``detections`` table by
``curation.ingest`` (class_label = 'crosswalk').
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..query import Query
from ..types import QueryOutput, FrameResult

from curation.database import get_connection


class CrosswalkCountQuery(Query):
    name = "Crosswalk Count"
    description = "Frames with N+ crosswalk detections above a confidence threshold"

    def build_params(self):
        return {
            "min_count": st.sidebar.slider("Min crosswalks", 1, 10, 1,
                                            key="cw_n"),
            "min_conf": st.sidebar.slider("Min confidence", 0.0, 1.0, 0.3,
                                           0.05, key="cw_c"),
            "max_results": st.sidebar.slider("Max frames", 50, 2000, 200,
                                              step=50, key="cw_max"),
        }

    def execute(self, root, segments, params):
        db_path = params.get("db_path", "")
        if not db_path or not Path(db_path).exists():
            return QueryOutput([], "table", "Crosswalk Count",
                               "No curation DB found.")

        min_conf = params["min_conf"]
        min_count = params["min_count"]
        max_results = params["max_results"]

        # The (class_label, confidence) index makes this selective and fast.
        conn = get_connection(db_path, readonly=True)
        rows = conn.execute(
            """SELECT s.name AS segment, fa.frame_index,
                      d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2,
                      d.confidence
               FROM detections d
               JOIN frame_annotations fa ON d.frame_ann_id = fa.frame_ann_id
               JOIN segments s ON fa.segment_id = s.segment_id
               WHERE d.class_label = 'crosswalk' AND d.confidence >= ?""",
            (min_conf,),
        ).fetchall()
        conn.close()

        # Restrict to the sidebar segment filter (glob applied upstream).
        seg_set = set(segments)

        grouped: dict[tuple[str, int], list[dict]] = {}
        for r in rows:
            seg = r["segment"]
            if seg_set and seg not in seg_set:
                continue
            key = (seg, r["frame_index"])
            grouped.setdefault(key, []).append({
                "bbox": [r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]],
                "confidence": r["confidence"],
            })

        out: list[FrameResult] = []
        for (seg, fidx), dets in grouped.items():
            if len(dets) < min_count:
                continue
            out.append(FrameResult(
                seg, f"{fidx:06d}",
                score=len(dets),
                metadata={"crosswalks": dets},
            ))

        out.sort(key=lambda x: (-x.score, -max(d["confidence"] for d in x.metadata["crosswalks"])))
        out = out[:max_results]

        return QueryOutput(
            out, "detection",
            f"Frames with >= {min_count} crosswalks (conf >= {min_conf})",
            f"{len(out)} frames",
        )
