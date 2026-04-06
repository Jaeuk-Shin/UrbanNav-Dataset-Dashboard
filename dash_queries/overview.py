"""Per-segment summary table query."""

import numpy as np

from dash_query import Query
from dash_types import QueryOutput, SegmentResult
from dash_loaders import load_json, load_poses, DATASET_FPS


class DatasetOverview(Query):
    name = "Dataset Overview"
    description = "Per-segment summary table (frames, pedestrians, velocity, distance)"

    def build_params(self):
        return {}

    def execute(self, root, segments, params):
        rows = []
        for seg in segments:
            r = {"segment": seg, "date": "-".join(seg.split("-")[:3])}
            rgb = root / "rgb" / seg
            if rgb.exists():
                r["frames"] = len(list(rgb.glob("*.jpg")))
            df = root / "annotations" / seg / "detections.json"
            if df.exists():
                dets = load_json(str(df))
                c = [d["pedestrian_count"] for d in dets.values()]
                r["max_peds"] = max(c, default=0)
                r["avg_peds"] = round(float(np.mean(c)), 1) if c else 0
            pf = root / "pose" / f"{seg}.txt"
            if pf.exists():
                p = load_poses(str(pf))
                if len(p) >= 2:
                    d = np.linalg.norm(np.diff(p[:, [2, 4]], axis=0), axis=1)
                    r["avg_vel_ms"] = round(float(np.mean(d) * DATASET_FPS), 3)
                    r["dist_m"] = round(float(np.sum(d)), 2)
            rows.append(r)
        return QueryOutput(
            [SegmentResult(r["segment"], metadata=r) for r in rows],
            "table", "Dataset Overview", f"{len(segments)} segments",
        )
