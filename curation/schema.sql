-- Dataset curation schema
-- Stores segment inventory, file paths, filter metrics, and split assignments.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------
-- Source videos (one YouTube video = many segments)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS videos (
    video_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE
);

-- ----------------------------------------------------------------
-- Segments (clips extracted from a video)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS segments (
    segment_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        INTEGER NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    segment_index   INTEGER NOT NULL,
    name            TEXT    NOT NULL UNIQUE,   -- e.g. "VideoTitle_0003"
    pose_path       TEXT,
    feature_path    TEXT,
    rgb_path        TEXT,
    annotation_dir  TEXT,
    num_frames      INTEGER NOT NULL,          -- rows in the pose file
    split           TEXT CHECK (split IN ('train', 'val', 'test')),
    UNIQUE (video_id, segment_index)
);
CREATE INDEX IF NOT EXISTS idx_segments_split ON segments(split);
CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);

-- ----------------------------------------------------------------
-- Per-frame annotations (one row per annotated frame).
-- Annotations are sparse: only a subset of frames have detections
-- (e.g. every ~150 frames in YouTube segments).
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS frame_annotations (
    frame_ann_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id       INTEGER NOT NULL REFERENCES segments(segment_id) ON DELETE CASCADE,
    frame_index      INTEGER NOT NULL,   -- frame index within the segment
    pedestrian_count INTEGER NOT NULL DEFAULT 0,
    crosswalk_count  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (segment_id, frame_index)
);
CREATE INDEX IF NOT EXISTS idx_frame_ann_segment
    ON frame_annotations(segment_id);
CREATE INDEX IF NOT EXISTS idx_frame_ann_segment_frame
    ON frame_annotations(segment_id, frame_index);

-- ----------------------------------------------------------------
-- Individual detections (pedestrians and crosswalks).
-- Normalised out of frame_annotations so that bounding boxes and
-- confidence values are queryable without parsing JSON blobs.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_ann_id INTEGER NOT NULL
                 REFERENCES frame_annotations(frame_ann_id) ON DELETE CASCADE,
    class_label  TEXT NOT NULL,   -- 'pedestrian' or 'crosswalk'
    confidence   REAL NOT NULL,
    bbox_x1      REAL,
    bbox_y1      REAL,
    bbox_x2      REAL,
    bbox_y2      REAL
);
CREATE INDEX IF NOT EXISTS idx_detections_frame
    ON detections(frame_ann_id);
CREATE INDEX IF NOT EXISTS idx_detections_class_conf
    ON detections(class_label, confidence);

-- ----------------------------------------------------------------
-- Pre-parsed pose trajectories (binary BLOBs).
-- Eliminates repeated np.loadtxt text parsing.  Each BLOB is a
-- contiguous C-order float64 array of shape (num_frames, 7) where
-- columns are [x, y, z, qx, qy, qz, qw].  The first column
-- (frame index) from the original text file is NOT stored.
-- Reconstruct with:
--   np.frombuffer(blob, dtype=np.float64).reshape(-1, 7)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS segment_poses (
    segment_id INTEGER PRIMARY KEY
                       REFERENCES segments(segment_id) ON DELETE CASCADE,
    pose_data  BLOB NOT NULL    -- float64 (N, 7) C-order
);

-- ----------------------------------------------------------------
-- Per-frame filter metrics (numpy arrays serialised as BLOBs)
-- One row per segment.  Each BLOB is a 1-D float32 numpy array of
-- length num_frames, serialised with ndarray.tobytes().
-- valid_mask is a bool (uint8) array: True = frame passes all filters.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS segment_filter_data (
    segment_id              INTEGER PRIMARY KEY
                            REFERENCES segments(segment_id) ON DELETE CASCADE,
    velocities              BLOB,   -- float32 (N,)  m/step
    forward_camera_angles   BLOB,   -- float32 (N,)  degrees [0,180]
    roll_changes            BLOB,   -- float32 (N,)  degrees per frame
    pitch_changes           BLOB,   -- float32 (N,)  degrees per frame
    yaw_changes             BLOB,   -- float32 (N,)  degrees per frame
    abs_pitch               BLOB,   -- float32 (N,)  degrees from horizontal
    abs_roll                BLOB,   -- float32 (N,)  degrees from level
    height_changes          BLOB,   -- float32 (N,)  |Δy|/|Δxyz| ratio [0,1]
    valid_mask              BLOB    -- uint8   (N,)  combined pass/fail
);

-- ----------------------------------------------------------------
-- Metadata for cached lookup tables
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lut_cache (
    lut_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,  -- e.g. "train_cs5_wp5"
    split        TEXT    NOT NULL,
    context_size INTEGER NOT NULL,
    wp_length    INTEGER NOT NULL,
    pose_step    INTEGER NOT NULL DEFAULT 1,
    filter_cfg   TEXT    NOT NULL,         -- JSON of filter thresholds
    file_path    TEXT    NOT NULL,         -- path to the .npz file
    num_entries  INTEGER NOT NULL,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);
