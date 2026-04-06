# OMR Dataset Explorer Dashboard

A plugin-based Streamlit dashboard for exploring the OMR outdoor mobile robot
dataset.  Queries filter data; visualizers render the results.  Both are
independent extension points -- adding a new query or a new visualization
requires **one new file** and **one line** in the corresponding registry.

## Quickstart

```bash
pip install streamlit plotly opencv-python numpy pandas Pillow
cd /raid/robot/real_world_dataset/omr
streamlit run dashboard.py
```

## Package layout

```
dashboard.py                        # entry point (streamlit run dashboard.py)
dashboard/
    __init__.py                     # re-exports Query, FrameResult, SegmentResult, QueryOutput
    types.py                        # result dataclasses
    query.py                        # Query abstract base class
    loaders.py                      # cached helpers: load_json, load_poses, list_segments
    app.py                          # Streamlit main()
    queries/
        __init__.py                 # QUERIES registry (list)
        overview.py                 # DatasetOverview
        pedestrian.py               # PedestrianCountQuery
        object_presence.py          # ObjectPresenceQuery
        ego_velocity.py             # EgoVelocityQuery
        caption_search.py           # CaptionSearchQuery
    visualizers/
        __init__.py                 # VISUALIZERS registry (dict)
        _common.py                  # PALETTE, load_rgb()
        image_grid.py               # vis_image_grid
        detection.py                # vis_detection
        mask.py                     # vis_mask
        trajectory.py               # vis_trajectory
        table.py                    # vis_table
```

## Architecture

```
┌──────────┐   build_params()   ┌────────────┐   execute()   ┌─────────────┐
│ Sidebar  │ ────────────────►  │   Query    │ ───────────►  │ QueryOutput │
│ widgets  │   (returns dict)   │ subclass   │  (filtering)  │  .viz_type  │
└──────────┘                    └────────────┘               └──────┬──────┘
                                                                    │
                                    VISUALIZERS[viz_type]           │
                                                                    ▼
                                                             ┌────────���────┐
                                                             │ Visualizer  │
                                                             │  function   │
                                                             └─────────────┘
```

1. The user picks a **Query** from the sidebar.
2. The query's `build_params()` renders its own sidebar widgets and returns a
   `dict` of parameter values.
3. Clicking **Run Query** calls `execute(root, segments, params)` which scans
   the dataset and returns a `QueryOutput`.
4. The `QueryOutput.viz_type` string selects a **visualizer** function from the
   `VISUALIZERS` dict, which renders the results.

Queries and visualizers are fully decoupled: any query can target any visualizer
by setting `viz_type`, and the same visualizer can render results from different
queries.

## Result types

All types live in `dashboard/types.py`.

### `FrameResult`

Represents a single frame hit.

| Field      | Type   | Description |
|------------|--------|-------------|
| `segment`  | `str`  | Segment name, e.g. `"2026-03-05-15-44-53_seg003"` |
| `frame_id` | `str`  | Zero-padded frame identifier, e.g. `"000030"` |
| `score`    | `float`| Ranking score (higher = more relevant). Default `0.0`. |
| `metadata` | `dict` | Arbitrary payload consumed by the visualizer. |

### `SegmentResult`

Represents a whole-segment hit (e.g. velocity query).

| Field      | Type   | Description |
|------------|--------|-------------|
| `segment`  | `str`  | Segment name. |
| `score`    | `float`| Ranking score. |
| `metadata` | `dict` | Arbitrary payload consumed by the visualizer. |

### `QueryOutput`

Returned by `Query.execute()`.

| Field         | Type   | Description |
|---------------|--------|-------------|
| `results`     | `list` | List of `FrameResult` or `SegmentResult`. |
| `viz_type`    | `str`  | Key into the `VISUALIZERS` dict (e.g. `"detection"`). |
| `title`       | `str`  | Heading displayed above the results. |
| `description` | `str`  | Subtitle / summary line (e.g. `"42 frames"`). |

## Built-in queries

| Name              | Result type     | viz_type      | What it does |
|-------------------|-----------------|---------------|--------------|
| Dataset Overview  | `SegmentResult` | `table`       | Per-segment stats table (frames, pedestrians, velocity, distance) with CSV download |
| Pedestrian Count  | `FrameResult`   | `detection`   | Frames with N+ pedestrians above a confidence threshold |
| Object in Masks   | `FrameResult`   | `mask`        | Frames containing a named object in segmentation masks |
| Ego Velocity      | `SegmentResult` | `trajectory`  | Segments whose mean XZ-plane velocity exceeds a threshold |
| Caption Search    | `FrameResult`   | `image_grid`  | Keyword search over frame captions and tags |

## Built-in visualizers

| viz_type       | Expected results | Renders |
|----------------|------------------|---------|
| `image_grid`   | `FrameResult`    | 4-column image thumbnails with caption/tag text |
| `detection`    | `FrameResult`    | Images with pedestrian bounding boxes overlaid |
| `mask`         | `FrameResult`    | Images with coloured mask overlay + labelled bboxes |
| `trajectory`   | `SegmentResult`  | Side-by-side XZ trajectory plot (colour = velocity) and velocity profile |
| `table`        | either           | Sortable pandas DataFrame + CSV download button |

## Shared data loaders

`dashboard/loaders.py` provides three `@st.cache_data` helpers that any query
can import.  They cache results across Streamlit reruns.

```python
from dashboard.loaders import load_json, load_poses, list_segments, DATASET_FPS
```

| Function | Signature | Returns |
|----------|-----------|---------|
| `load_json(path)` | `str -> dict` | Parsed JSON (cached by path) |
| `load_poses(path)` | `str -> np.ndarray` | Pose array, shape `(N, 9)`: idx, frame_id, x, y, z, qx, qy, qz, qw |
| `list_segments(rgb_dir)` | `str -> list[str]` | Sorted segment directory names |

`DATASET_FPS = 2.0` is the approximate frame rate of the dataset (original
30 Hz camera subsampled ~15x).

## Visualizer helpers

`dashboard/visualizers/_common.py` provides:

| Name | Description |
|------|-------------|
| `PALETTE` | List of 12 distinct RGB tuples for bounding boxes / mask overlays |
| `load_rgb(root, seg, fid)` | Load `root/rgb/{seg}/{fid}.jpg` as an `(H, W, 3)` uint8 numpy array, or `None` if missing |

---

## Writing a new query

### Step 1 -- Create the file

Create `dashboard/queries/my_query.py`:

```python
"""Example: find frames where a specific tag appears at least N times."""

import streamlit as st

from ..query import Query
from ..types import QueryOutput, FrameResult
from ..loaders import load_json


class TagCountQuery(Query):
    # ---- Class attributes (required) -----------------------------------------
    name = "Tag Count"                           # shown in the sidebar dropdown
    description = "Frames where a tag appears N+ times across nearby frames"

    # ---- build_params --------------------------------------------------------
    # Create Streamlit sidebar widgets for user-facing parameters.
    # Return a plain dict -- it will be passed to execute() as `params`.
    # IMPORTANT: every widget MUST have a unique `key` to avoid collisions
    # with widgets from other queries.

    def build_params(self) -> dict:
        return {
            "tag": st.sidebar.text_input("Tag", "car", key="tc_tag"),
            "min_count": st.sidebar.slider(
                "Min detections", 1, 50, 5, key="tc_min"
            ),
        }

    # ---- execute -------------------------------------------------------------
    # Scan segments, build a list of FrameResult or SegmentResult, and return
    # a QueryOutput whose viz_type selects the renderer.
    #
    # Arguments:
    #   root      -- Path to the dataset directory (contains rgb/, pose/,
    #                annotations/).
    #   segments  -- List of segment name strings already filtered by the
    #                user's glob pattern in the sidebar.
    #   params    -- The dict returned by build_params().

    def execute(self, root, segments, params):
        tag = params["tag"].lower().strip()
        if not tag:
            return QueryOutput([], "image_grid", "Enter a tag", "")

        results = []
        for seg in segments:
            cap_file = root / "annotations" / seg / "captions.json"
            if not cap_file.exists():
                continue
            caps = load_json(str(cap_file))

            # Count how many frames in this segment contain the tag
            matching_frames = []
            for fid, info in caps.items():
                if any(tag in t.lower() for t in info.get("tags", [])):
                    matching_frames.append(fid)

            if len(matching_frames) >= params["min_count"]:
                for fid in matching_frames:
                    results.append(FrameResult(
                        segment=seg,
                        frame_id=fid,
                        score=len(matching_frames),
                        metadata={
                            "caption": caps[fid].get("caption", ""),
                            "tags": caps[fid].get("tags", []),
                            "segment_tag_count": len(matching_frames),
                        },
                    ))

        results.sort(key=lambda r: -r.score)
        return QueryOutput(
            results=results,
            viz_type="image_grid",      # reuse the built-in image grid
            title=f'Segments with {params["min_count"]}+ "{tag}" frames',
            description=f"{len(results)} frames",
        )
```

Key points:

- **`name`** and **`description`** are mandatory class attributes.
- **`build_params()`** uses `st.sidebar.*` widgets and must give each widget
  a unique `key` string.  The returned dict is opaque to the framework -- its
  structure is a private contract between `build_params` and `execute`.
- **`execute()`** receives a `pathlib.Path` root, a pre-filtered list of
  segment names, and the params dict.  It returns a `QueryOutput`.
- **`viz_type`** can be any key in `VISUALIZERS`.  Reuse a built-in one, or
  create your own (see below).
- Frame-level queries return `FrameResult`; segment-level queries return
  `SegmentResult`.  The `metadata` dict carries whatever the visualizer needs.
- Use `load_json` / `load_poses` from `dashboard.loaders` for automatic
  Streamlit caching.

### Step 2 -- Register

Edit `dashboard/queries/__init__.py`:

```python
from .my_query import TagCountQuery          # add import

QUERIES: list = [
    DatasetOverview(),
    PedestrianCountQuery(),
    ObjectPresenceQuery(),
    EgoVelocityQuery(),
    CaptionSearchQuery(),
    TagCountQuery(),                         # add instance
]
```

That's it -- the new query appears in the sidebar dropdown on the next reload.

### Metadata conventions

The `metadata` dict is a free-form contract between the query and the
visualizer it targets.  If you target a built-in visualizer, match the
metadata shape it expects:

| viz_type     | Expected `metadata` keys |
|--------------|--------------------------|
| `image_grid` | `caption` (str, optional), `tags` (list[str], optional) |
| `detection`  | `pedestrians` (list of `{"bbox": [x1,y1,x2,y2], "confidence": float}`) |
| `mask`       | `masks` (list of `{"mask_id": str, "label": str, "score": float, "bbox": [x1,y1,x2,y2]}`) |
| `trajectory` | `positions` (list of [x,y,z]), `velocities` (list of float), `avg_velocity` (float), `max_velocity` (float) |
| `table`      | Any flat dict -- each key becomes a column |

---

## Writing a new visualizer

### Step 1 -- Create the file

Create `dashboard/visualizers/histogram.py`:

```python
"""Example: render a histogram of result scores."""

from pathlib import Path

import plotly.express as px
import streamlit as st

from ..types import QueryOutput


def vis_histogram(output: QueryOutput, root: Path, max_n: int):
    """Every visualizer has this exact signature:

    Parameters
    ----------
    output : QueryOutput
        The full output returned by the query.
    root : Path
        Dataset root directory (contains rgb/, pose/, annotations/).
    max_n : int
        Maximum number of results to display (from the sidebar slider).
    """
    items = output.results[:max_n]
    if not items:
        st.info("No results.")
        return

    scores = [r.score for r in items]
    fig = px.histogram(x=scores, nbins=30, labels={"x": "Score"})
    fig.update_layout(title="Score distribution", height=400)
    st.plotly_chart(fig, use_container_width=True)
```

Helpers available from `_common.py`:

```python
from ._common import PALETTE, load_rgb

img = load_rgb(root, result.segment, result.frame_id)  # (H,W,3) or None
color = PALETTE[i % len(PALETTE)]                       # RGB tuple
```

### Step 2 -- Register

Edit `dashboard/visualizers/__init__.py`:

```python
from .histogram import vis_histogram               # add import

VISUALIZERS = {
    ...,
    "histogram": vis_histogram,                     # add entry
}
```

Now any query that sets `viz_type="histogram"` in its `QueryOutput` will
use this renderer.

---

## Dataset reference

The dataset root (default `/raid/robot/real_world_dataset/omr/dataset`)
contains three subdirectories that queries can read.

### `rgb/{segment}/`

Sequential JPEG frames: `000000.jpg` through `000240.jpg` (symlinks to
undistorted cam_front_l images, 1280 x 720).

### `pose/{segment}.txt`

Tab/space-delimited, one row per frame:

```
idx  frame_id  x  y  z  qx  qy  qz  qw
```

- Columns 2-4: position in metres.
- Columns 5-8: orientation as a unit quaternion.
- ~241 rows per segment at ~2 fps.

### `annotations/{segment}/`

| File | Format | Content |
|------|--------|---------|
| `detections.json` | `{ "000030": { "pedestrian_count": 5, "pedestrians": [{"bbox": [x1,y1,x2,y2], "confidence": 0.82}, ...] } }` | Pedestrian bounding boxes (every 10th frame) |
| `captions.json` | `{ "000020": { "caption": "...", "tags": ["building", "person"] } }` | Scene captions and semantic tags |
| `embeddings.npy` | `(25, 1152) float16` | SigLIP ViT-SO400M image embeddings |
| `masks/{frame}.json` | `{ "masks": { "1": {"label": "sidewalk", "score": 0.53, "bbox": [...]}, ... } }` | Per-object segmentation metadata |
| `masks/{frame}.png` | uint16 label map (0 = background) | Pixel-level segmentation mask |
| `masks_done.json` | `{ "000000": 2, ... }` | Mask counts per frame (resume marker) |

`annotations/embedding_meta.json` (at root level) records the embedding model
name and pretrained tag.

---

## Tips

- **Widget keys must be unique across all queries.**  Use a short prefix per
  query (e.g. `"tc_"` for TagCountQuery) to avoid collisions.
- **Reuse loaders.**  `load_json` and `load_poses` cache automatically via
  `@st.cache_data`.  Don't re-implement JSON/pose loading.
- **Reuse visualizers.**  Before writing a new renderer, check if a built-in
  one works -- `image_grid` and `table` are generic enough for many use cases.
- **`score` drives sort order.**  Results are typically sorted by descending
  `score` before being returned in `QueryOutput`.  The visualizer does not
  re-sort.
- **Annotations are sampled every 10 frames** (`000000`, `000010`, `000020`,
  ...).  RGB frames exist for every frame (`000000` through `000240`).
  Frame-level queries on annotation data will only match annotated frames.
- **Segment names** follow the pattern `YYYY-MM-DD-HH-MM-SS_segNNN`.  The
  sidebar glob filter narrows which segments are passed to `execute()`.
