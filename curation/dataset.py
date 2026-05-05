"""PyTorch Dataset that loads from a pre-built filtered LUT.

Drop-in replacement for ``CarlaFeatDataset``: the ``__getitem__`` return
dict has the same keys (``obs_features``, ``input_positions``, ``waypoints``,
``step_scale``, plus extra keys in val/test mode).

Usage::

    from curation.dataset import FilteredFeatDataset

    ds = FilteredFeatDataset(
        lut_path="lut_train.npz",
        cfg=cfg,       # same OmegaConf as CarlaFeatDataset
        mode="train",
    )
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm

from .poses import load_pose_from_blob, load_pose_from_text


class FilteredFeatDataset(Dataset):
    """Feature dataset backed by a pre-built filtered lookup table.

    Parameters
    ----------
    lut_path : str
        Path to the ``.npz`` produced by ``build_lut.py``.
    cfg : OmegaConf
        Full training config (same structure as ``CarlaFeatDataset``).
    mode : str
        ``'train'``, ``'val'``, or ``'test'``.
    db_path : str or None
        Path to the curation SQLite database.  When provided, poses are
        loaded from the pre-parsed ``segment_poses`` table (fast binary
        read) instead of re-parsing text files.
    """

    def __init__(self, lut_path: str, cfg, mode: str, db_path: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.context_size = cfg.model.obs_encoder.context_size
        self.wp_length = cfg.model.decoder.len_traj_pred
        self.video_fps = cfg.data.video_fps
        self.pose_fps = cfg.data.pose_fps
        self.target_fps = cfg.data.target_fps
        self.input_noise = cfg.data.input_noise

        self.pose_step = max(1, self.pose_fps // self.target_fps)
        self.frame_step = self.video_fps // self.target_fps

        # Load pre-built LUT
        data = np.load(lut_path, allow_pickle=True)
        self.lut = data["lut"]  # (M, 2) int32: (seg_local_idx, pose_start)
        segment_names = data["segment_names"]  # (S,) object
        segment_paths = data["segment_paths"]  # (S, 4) object
        self.video_ranges = data["video_ranges"]  # (S, 2) int32

        assert len(self.lut) > 0, f"LUT at {lut_path} is empty"

        # Unpack segment paths
        self.segment_names = list(segment_names)
        self.pose_paths = [str(p[0]) for p in segment_paths]
        self.feature_paths = [str(p[1]) for p in segment_paths]
        self.rgb_paths = [str(p[2]) for p in segment_paths]

        # Load metadata from feature directory
        if self.feature_paths:
            feat_dir = os.path.dirname(self.feature_paths[0])
            metadata_path = os.path.join(feat_dir, "metadata.pt")
            if os.path.exists(metadata_path):
                meta = torch.load(metadata_path, weights_only=True)
                self.feature_dim = meta["feature_dim"]
                self.include_flip = meta.get("include_flip", False)
            else:
                self.feature_dim = 768
                self.include_flip = False
        else:
            self.feature_dim = 768
            self.include_flip = False

        # Load poses (subsampled by pose_step).
        # Prefer binary BLOBs from the DB (fast) over text files (slow).
        self.poses = []
        self.step_scale = []

        db_conn = None
        seg_name_to_id: dict[str, int] = {}
        if db_path and os.path.exists(db_path):
            from .database import get_connection
            db_conn = get_connection(db_path, readonly=True)
            rows = db_conn.execute(
                "SELECT segment_id, name FROM segments"
            ).fetchall()
            seg_name_to_id = {r["name"]: r["segment_id"] for r in rows}

        for i, pp in enumerate(tqdm(self.pose_paths, desc="Loading poses")):
            pose = None
            seg_name = self.segment_names[i]
            seg_id = seg_name_to_id.get(seg_name)

            # Try DB first
            if db_conn is not None and seg_id is not None:
                row = db_conn.execute(
                    "SELECT pose_data FROM segment_poses WHERE segment_id = ?",
                    (seg_id,),
                ).fetchone()
                if row is not None:
                    full = load_pose_from_blob(row["pose_data"])
                    if full.shape[0] > 0:
                        pose = full[:: self.pose_step]

            # Fall back to text file
            if pose is None and pp:
                pose = load_pose_from_text(pp)[:: self.pose_step]

            self.poses.append(pose)
            ss = np.linalg.norm(np.diff(pose[:, [0, 2]], axis=0), axis=1).mean()
            self.step_scale.append(ss)

        if db_conn is not None:
            db_conn.close()

        # Camera intrinsics (optional)
        self._camera = self._parse_camera_intrinsics(cfg)

        # Augmentation
        self.augment = mode == "train"
        self.horizontal_flip_prob = (
            getattr(cfg.data, "horizontal_flip_prob", 0.5)
            if (self.augment and self.include_flip)
            else 0.0
        )

        # Per-worker feature cache
        self._feat_cache = {"idx": None, "data": None}

        # Video paths for decord visualisation
        _VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi", ".mov")
        self.image_dirs = []
        self.video_paths_list = []
        for rp in self.rgb_paths:
            if rp and os.path.isdir(rp):
                self.image_dirs.append(rp)
                self.video_paths_list.append(None)
            elif rp and os.path.isfile(rp):
                self.image_dirs.append(None)
                self.video_paths_list.append(rp)
            else:
                self.image_dirs.append(None)
                self.video_paths_list.append(None)

    # ------------------------------------------------------------------ #
    # Camera intrinsics (same as CarlaFeatDataset)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_camera_intrinsics(cfg):
        def _from_fov(width, height, fov, dw, dh):
            f = 0.5 * width / np.tan(float(fov) * np.pi / 360.0)
            return [f, f, 0.5 * width, 0.5 * height, dw, dh]

        cam = getattr(cfg.data, "camera", None)
        if cam is not None:
            dw = float(cam["desired_width"])
            dh = float(cam["desired_height"])
            if "fx" in cam:
                return [
                    float(cam["fx"]),
                    float(cam["fy"]),
                    float(cam["cx"]),
                    float(cam["cy"]),
                    dw,
                    dh,
                ]
            else:
                return _from_fov(
                    float(cam["width"]),
                    float(cam["height"]),
                    cam["fov"],
                    dw,
                    dh,
                )

        keys = ("width", "height", "fov", "desired_width", "desired_height")
        if all(hasattr(cfg.data, k) for k in keys):
            d = cfg.data
            return _from_fov(
                float(d.width),
                float(d.height),
                d.fov,
                float(d.desired_width),
                float(d.desired_height),
            )
        return None

    # ------------------------------------------------------------------ #
    # Dataset interface                                                  #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.lut)

    def _load_features(self, video_idx: int):
        if self._feat_cache["idx"] != video_idx:
            self._feat_cache["data"] = torch.load(
                self.feature_paths[video_idx], weights_only=True
            )
            self._feat_cache["idx"] = video_idx
        return self._feat_cache["data"]

    def __getitem__(self, index):
        video_idx, pose_start = int(self.lut[index, 0]), int(self.lut[index, 1])
        frame_indices = self.frame_step * np.arange(
            pose_start, pose_start + self.context_size
        )

        # Load cached features
        feat_data = self._load_features(video_idx)
        features = feat_data["features"]
        num_frames = features.shape[0]
        frame_indices = [min(idx, num_frames - 1) for idx in frame_indices]

        use_flip = (
            self.augment
            and self.include_flip
            and random.random() < self.horizontal_flip_prob
        )
        if use_flip:
            obs_features = feat_data["features_flip"][frame_indices]
        else:
            obs_features = features[frame_indices]

        # --- Pose processing (identical to CarlaFeatDataset) ---
        pose = self.poses[video_idx]
        input_poses = pose[pose_start: pose_start + self.context_size]
        original_input_poses = np.copy(input_poses)

        waypoint_poses = pose[
            pose_start + self.context_size:
            pose_start + self.context_size + self.wp_length
        ]

        if self.input_noise > 0:
            input_poses = self._add_noise(input_poses)

        current_pose = input_poses[-1]

        if self.cfg.model.cord_embedding.type == "input_target":
            transformed_input_positions = self._transform_poses(
                input_poses, current_pose
            )[:, [0, 2]]
        else:
            raise NotImplementedError(
                f"Coordinate embedding type {self.cfg.model.cord_embedding.type} "
                f"not implemented"
            )

        waypoints_transformed = self._transform_waypoints(waypoint_poses, current_pose)
        waypoint_transformed_y = np.copy(waypoints_transformed[:, 1])

        input_positions = torch.tensor(
            transformed_input_positions, dtype=torch.float32
        )
        waypoints_transformed = torch.tensor(
            waypoints_transformed[:, [0, 2]], dtype=torch.float32
        )
        step_scale = torch.tensor(self.step_scale[video_idx], dtype=torch.float32)
        step_scale = torch.clamp(step_scale, min=1e-2)
        input_positions_scaled = input_positions / step_scale
        waypoints_scaled = waypoints_transformed / step_scale
        input_positions_scaled[: self.context_size - 1] += (
            torch.randn(self.context_size - 1, 2) * self.input_noise
        )

        if use_flip:
            input_positions_scaled[:, 0] *= -1
            waypoints_scaled[:, 0] *= -1

        sample = {
            "obs_features": obs_features,
            "input_positions": input_positions_scaled,
            "waypoints": waypoints_scaled,
            "step_scale": step_scale,
        }

        if self.mode in ["val", "test"]:
            transformed_original = self._transform_poses(
                original_input_poses, current_pose
            )
            original_input_positions = torch.tensor(
                transformed_original[:, [0, 2]], dtype=torch.float32
            )
            noisy_input_positions = input_positions_scaled[:-1] * step_scale

            sample["original_input_positions"] = original_input_positions
            sample["noisy_input_positions"] = noisy_input_positions
            sample["gt_waypoints"] = waypoints_transformed
            sample["gt_waypoints_y"] = waypoint_transformed_y

            image_path = ""
            if self.image_dirs[video_idx] is not None:
                dirpath = self.image_dirs[video_idx]
                img_files = sorted(
                    f for f in os.listdir(dirpath) if f.endswith(".jpg")
                )
                if img_files:
                    last_idx = min(frame_indices[-1], len(img_files) - 1)
                    image_path = os.path.join(dirpath, img_files[last_idx])
            sample["image_path"] = image_path

            video_path = ""
            video_frame_idx = -1
            if not image_path and self.video_paths_list[video_idx] is not None:
                video_path = self.video_paths_list[video_idx]
                video_frame_idx = int(frame_indices[-1])
            sample["video_path"] = video_path
            sample["video_frame_idx"] = video_frame_idx

            if self._camera is not None:
                sample["camera_intrinsics"] = torch.tensor(
                    self._camera, dtype=torch.float32
                )
            else:
                sample["camera_intrinsics"] = torch.full((6,), -1.0)

        return sample

    # ------------------------------------------------------------------ #
    # Helpers (same as CarlaFeatDataset)                                 #
    # ------------------------------------------------------------------ #

    def _add_noise(self, input_poses):
        noise = np.random.normal(0, self.input_noise, input_poses[:, :3].shape)
        scale = np.linalg.norm(input_poses[-1, :3] - input_poses[-2, :3])
        input_poses = np.copy(input_poses)
        input_poses[:, :3] += noise * scale
        return input_poses

    def _transform_poses(self, poses, current_pose_array):
        current_pose_matrix = self._pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrices = self._poses_to_matrices(poses)
        transformed = np.matmul(
            current_pose_inv[np.newaxis, :, :], pose_matrices
        )
        return transformed[:, :3, 3]

    def _transform_waypoints(self, waypoint_poses, current_pose_array):
        current_pose_matrix = self._pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        waypoint_matrices = self._poses_to_matrices(waypoint_poses)
        transformed = np.matmul(
            current_pose_inv[np.newaxis, :, :], waypoint_matrices
        )
        return transformed[:, :3, 3]

    def _pose_to_matrix(self, pose):
        position = pose[:3]
        rotation = R.from_quat(pose[3:])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = position
        return matrix

    def _poses_to_matrices(self, poses):
        positions = poses[:, :3]
        quats = poses[:, 3:]
        rotations = R.from_quat(quats)
        matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
        matrices[:, :3, :3] = rotations.as_matrix()
        matrices[:, :3, 3] = positions
        return matrices
