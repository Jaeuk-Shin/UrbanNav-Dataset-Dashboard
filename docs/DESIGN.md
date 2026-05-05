# Database Design for Heterogeneous CV/ML Datasets

This document captures a design conversation covering relational database schema,
temporal frame ordering, heterogeneous data sources, batching strategies, and
binary data storage for a computer vision / deep learning pipeline.

---

## 1. Core Schema: Normalization and the Flat Table Problem

A naive "flat" dataframe representation repeats frame-level fields (image path,
odometry, etc.) N times per frame where N is the number of detected objects.
This violates database normalization and causes redundancy, update anomalies,
and inefficient memory usage.

### Solution: Decompose into Related Tables

**`videos` table** — one row per video/session source:

```sql
CREATE TABLE videos (
    video_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,  -- file path, device id, stream url, etc.
    fps        REAL NOT NULL,
    width      INTEGER,
    height     INTEGER
);
```

**`frames` table** — one row per frame:

```sql
CREATE TABLE frames (
    frame_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    INTEGER NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    timestamp   REAL,               -- seconds from video start
    image_path  TEXT,
    odom_x      REAL,
    odom_y      REAL,
    odom_theta  REAL,
    UNIQUE (video_id, frame_index)
);
```

**`detections` table** — one row per detected object:

```sql
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
    class_label  TEXT    NOT NULL,
    bbox_x       REAL,
    bbox_y       REAL,
    bbox_w       REAL,
    bbox_h       REAL,
    confidence   REAL
);

CREATE INDEX idx_detections_frame_id ON detections(frame_id);
```

Reconstruct full data with a JOIN:

```sql
SELECT f.image_path, f.odom_x, f.odom_y, d.class_label, d.bbox_x, d.confidence
FROM frames f
JOIN detections d ON f.frame_id = d.frame_id
WHERE f.frame_id = 1;
```

---

## 2. Temporal Ordering of Frames

### Recommended: frame_index + fps

Store `frame_index` (integer) in the `frames` table and `fps` once in the
`videos` table. Derive timestamp on query:

```sql
SELECT f.*, (f.frame_index / v.fps) AS timestamp_sec
FROM frames f
JOIN videos v ON f.video_id = v.video_id
WHERE f.video_id = 1
  AND f.frame_index BETWEEN 42 AND 46;
```

### For Event-Driven Sources (ROS bags, variable-rate sensors)

Store an explicit `timestamp` per frame and index on `(session_id, timestamp)`:

```sql
CREATE INDEX idx_frames_session_time ON frames(session_id, timestamp);

-- 5 frames nearest to t=12.5s
SELECT * FROM frames
WHERE session_id = 1
  AND timestamp >= 12.5
ORDER BY timestamp
LIMIT 5;
```

### Avoid: Linked List (next_frame_id pointer)

Self-referential FK requiring recursive CTEs — no random access, no range
queries, poor query planner support. Do not use for sequential frame data.

---

## 3. Heterogeneous Data Sources (Class Table Inheritance)

When data comes from mixed sources (e.g. YouTube fixed-fps video and ROS bags),
use **Class Table Inheritance**: a shared base `frames` table plus per-source
extension tables. Materialize `timestamp` into the base table at insert time
so all downstream queries are uniform.

```sql
-- Universal frame attributes
CREATE TABLE frames (
    frame_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    image_path  TEXT,
    odom_x      REAL,
    odom_y      REAL,
    odom_theta  REAL,
    timestamp   REAL NOT NULL   -- always populated at insert time
);

-- Sources table
CREATE TABLE sources (
    source_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL CHECK (source_type IN ('youtube', 'rosbag')),
    label       TEXT,
    fps         REAL,       -- YouTube only
    url         TEXT,       -- YouTube only
    bag_path    TEXT,       -- ROS only
    ros_topic   TEXT        -- ROS only
);

-- YouTube-specific extension
CREATE TABLE frame_meta_youtube (
    frame_id    INTEGER PRIMARY KEY REFERENCES frames(frame_id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL
);

-- ROS-specific extension
CREATE TABLE frame_meta_rosbag (
    frame_id  INTEGER PRIMARY KEY REFERENCES frames(frame_id) ON DELETE CASCADE,
    ros_seq   INTEGER,
    ros_topic TEXT
);
```

All temporal queries are now uniform regardless of source type:

```sql
SELECT f.*, d.*
FROM frames f
JOIN detections d ON f.frame_id = d.frame_id
WHERE f.source_id = :source_id
  AND f.timestamp BETWEEN :t_start AND :t_end
ORDER BY f.timestamp
LIMIT 5;
```

`detections` references the base `frames` table — no ambiguity.

---

## 4. Train/Val/Test Splits Table

```sql
CREATE TABLE splits (
    frame_id INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
    split    TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    PRIMARY KEY (frame_id, split)
);
```

---

## 5. Batching Strategy for Deep Learning

### Do NOT use ORDER BY RANDOM()

```sql
-- NEVER do this in a training loop — O(N log N) full table scan
SELECT * FROM frames ORDER BY RANDOM() LIMIT 32;
```

### Standard Two-Stage Architecture

**Stage 1 — Build index manifest at startup (one query):**

```python
class FrameDataset:
    def __init__(self, db_path, split='train'):
        conn = sqlite3.connect(db_path)
        rows = conn.execute("""
            SELECT f.frame_id, f.source_id, f.timestamp, f.image_path
            FROM frames f
            JOIN splits s ON f.frame_id = s.frame_id
            WHERE s.split = ?
            ORDER BY f.frame_id
        """, (split,)).fetchall()
        conn.close()
        self.index = np.array([r[0] for r in rows])
        self.meta  = {r[0]: r[1:] for r in rows}
```

**Stage 2 — Sampler operates purely in memory:**

```python
class RandomSampler:
    def __init__(self, dataset, batch_size, seed=42):
        self.index      = dataset.index.copy()
        self.batch_size = batch_size
        self.rng        = np.random.default_rng(seed)

    def __iter__(self):
        shuffled = self.rng.permutation(self.index)
        for i in range(0, len(shuffled), self.batch_size):
            yield shuffled[i : i + self.batch_size]
```

**Stage 3 — Workers fetch by explicit IDs (never RANDOM()):**

```python
class FrameDataset(torch.utils.data.Dataset):
    def __getitem__(self, frame_id):
        conn = self._get_connection()
        placeholders = ','.join('?' * len(frame_ids))
        rows = conn.execute(f"""
            SELECT f.*, d.class_label, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, d.confidence
            FROM frames f
            LEFT JOIN detections d ON f.frame_id = d.frame_id
            WHERE f.frame_id IN ({placeholders})
        """, frame_ids).fetchall()
        return self._parse_rows(rows)

    def _get_connection(self):
        tid = threading.get_ident()
        if tid not in self._conns:
            self._conns[tid] = sqlite3.connect(self._db_path)
        return self._conns[tid]
```

**PyTorch DataLoader wiring:**

```python
dataset    = FrameDataset(db_path='data.db')
sampler    = RandomSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_sampler  = BatchSampler(sampler, batch_size=32, drop_last=True),
    num_workers    = 4,
    prefetch_factor= 2,
    pin_memory     = True,
)
```

**SQLite multi-worker settings:**

```sql
PRAGMA journal_mode=WAL;   -- allows concurrent reads
PRAGMA foreign_keys=ON;
```

---

## 6. LMDB for Raw Image/Tensor Storage

SQLite handles metadata and annotations. LMDB handles raw binary blobs
(images, tensors) via memory-mapped I/O — no memcpy, no userspace buffer,
OS page cache is the database cache.

### Writing

```python
import lmdb

env = lmdb.open('frames.lmdb', map_size=1 << 40)  # 1TB virtual (sparse)

with env.begin(write=True) as txn:
    for frame_id, image in frames:
        key   = f'frame_{frame_id:08d}'.encode()
        value = image.tobytes()
        txn.put(key, value)
```

### Reading (zero-copy)

```python
with env.begin(write=False) as txn:
    buf = txn.get(b'frame_00000042')
    img = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, C)
```

### Combined Dataset class

```python
class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, lmdb_path):
        conn = sqlite3.connect(db_path)
        self.index       = conn.execute(
            "SELECT frame_id FROM frames WHERE split='train'"
        ).fetchall()
        self.annotations = {}  # populated from detections table
        conn.close()
        self.lmdb_path = lmdb_path
        self._env = None

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly   =True,
                lock       =False,
                readahead  =False,
                meminit    =False,
            )
        return self._env

    def __getitem__(self, frame_id):
        with self._get_env().begin() as txn:
            buf = txn.get(f'frame_{frame_id:08d}'.encode())
            img = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, C)
        ann = self.annotations[frame_id]
        return img, ann
```

---

## 7. Storage Decision Guide

```
Raw images/tensors (large binary blobs, random access)?
└── LMDB

Metadata, annotations, splits, temporal queries?
└── SQLite

Dataset fits entirely in RAM?
└── Load into memory at startup, skip LMDB

Sequential access pattern (streaming)?
└── HDF5 or Zarr

Multi-process write workload?
└── PostgreSQL
```

---

## Implementation Task

Based on the design above, implement the following files:

### `schema.sql`
SQLite schema with all tables:
- `sources` (with source_type CHECK constraint)
- `videos` (for fixed-fps sources)
- `frames` (base table, always has materialized `timestamp`)
- `frame_meta_youtube` (extension: frame_index)
- `frame_meta_rosbag` (extension: ros_seq, ros_topic)
- `detections` (with FK to frames, indexed on frame_id)
- `splits` (train/val/test assignment)
- All necessary indexes
- WAL mode and foreign key PRAGMAs

### `db.py`
Database connection and query helpers:
- `get_connection(db_path, readonly=False)` with WAL + FK pragmas
- `get_frames_in_window(conn, source_id, t_start, t_end)` — uniform temporal query
- `get_detections_for_frame(conn, frame_id)`
- `insert_frame(conn, source_id, source_type, image_path, odom, timestamp, meta)`

### `ingest.py`
Ingestion scripts for each source type:
- `ingest_youtube(db_path, video_path, fps, source_label)` — extracts frames,
  computes `timestamp = frame_index / fps`, writes to LMDB, inserts into DB
- `ingest_rosbag(db_path, bag_path, topic)` — reads ROS bag, uses header
  timestamp directly, writes to LMDB, inserts into DB

### `lmdb_store.py`
LMDB read/write utilities:
- `LMDBWriter(lmdb_path, map_size)` — context manager, batched writes
- `LMDBReader(lmdb_path)` — thread-local env, zero-copy read returning np.ndarray
- Key convention: `frame_{frame_id:08d}`

### `dataset.py`
PyTorch Dataset + Sampler:
- `FrameDataset(db_path, lmdb_path, split)` — loads index from SQLite at init,
  reads tensors from LMDB in `__getitem__`
- `RandomSampler(dataset, batch_size, seed)` — pure in-memory permutation
- `TemporalWindowSampler(dataset, window_size, batch_size)` — samples
  contiguous windows of frames from the same source
- Thread-local SQLite connections in DataLoader workers

### `README.md`
Brief usage examples for ingestion and training loop setup.
