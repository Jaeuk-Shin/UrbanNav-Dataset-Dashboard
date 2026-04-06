# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit dashboard for exploring the OMR outdoor mobile robot dataset. Plugin-based architecture: **queries** filter data, **visualizers** render results. Both are independent extension points.

## Running

```bash
cd /raid/robot/real_world_dataset/omr
streamlit run dashboard.py
```

Dependencies: `streamlit plotly opencv-python numpy pandas Pillow`

The entry point is `dashboard.py` (parent directory), which calls `dashboard/app.py:main()`. The default dataset root is `/raid/robot/real_world_dataset/omr/dataset`.

## Architecture

Query-visualizer pipeline, fully decoupled:

1. User selects a **Query** subclass from the sidebar dropdown (`queries/__init__.py:QUERIES` list)
2. `Query.build_params()` renders sidebar widgets, returns a `dict`
3. `Query.execute(root, segments, params)` scans the dataset, returns a `QueryOutput`
4. `QueryOutput.viz_type` string selects a function from `visualizers/__init__.py:VISUALIZERS` dict
5. The visualizer function renders results with signature `(output: QueryOutput, root: Path, max_n: int)`

Any query can target any visualizer via `viz_type`. The `metadata` dict on each result is a free-form contract between the query and its target visualizer.

## Registries

- **Queries**: `queries/__init__.py` — `QUERIES` list of instantiated `Query` subclasses
- **Visualizers**: `visualizers/__init__.py` — `VISUALIZERS` dict mapping `viz_type` string to function

Adding a new query or visualizer: create one file, add one import + registration line in the corresponding `__init__.py`.

## Key Types (`types.py`)

- `FrameResult(segment, frame_id, score, metadata)` — single-frame hit
- `SegmentResult(segment, score, metadata)` — whole-segment hit
- `QueryOutput(results, viz_type, title, description)` — returned by `execute()`

## Conventions

- **Widget keys must be globally unique** across all queries. Use a short prefix per query (e.g. `"pc_"` for PedestrianCountQuery).
- **Use `loaders.py` helpers** (`load_json`, `load_poses`, `list_segments`) — they are `@st.cache_data` wrapped. Don't re-implement JSON/pose loading.
- **`score` drives sort order** — queries sort results by descending score before returning. Visualizers do not re-sort.
- **Annotations are sampled every 10 frames** (`000000`, `000010`, `000020`, ...). RGB frames exist for every frame. Frame-level queries on annotations only match annotated frames.
- `DATASET_FPS = 2.0` — approximate frame rate (original 30 Hz subsampled ~15x).

## Metadata Shape by Visualizer

When targeting a built-in visualizer, the query's `metadata` dict must match:

| viz_type | Expected keys |
|---|---|
| `image_grid` | `caption` (str), `tags` (list[str]) — both optional |
| `detection` | `pedestrians` (list of `{bbox, confidence}`) |
| `mask` | `masks` (list of `{mask_id, label, score, bbox}`) |
| `trajectory` | `positions`, `velocities`, `avg_velocity`, `max_velocity` |
| `table` | any flat dict — each key becomes a column |

## Shared Helpers

- `loaders.py`: `load_json(path)`, `load_poses(path) -> ndarray(N,9)`, `list_segments(rgb_dir)`
- `visualizers/_common.py`: `PALETTE` (12 RGB tuples), `load_rgb(root, seg, fid) -> ndarray|None`
