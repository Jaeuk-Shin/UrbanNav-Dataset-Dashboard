"""Lightning DataModule that uses pre-built filtered LUTs.

Drop-in replacement for ``UrbanNavFeatMixtureDataModule``.  Each mixture
entry can specify ``lut_train`` / ``lut_val`` / ``lut_test`` paths pointing
to cached ``.npz`` files produced by ``build_lut.py``.  Entries without
LUT paths fall back to the original ``CarlaFeatDataset``.

Example config entry::

    mixture:
      - root: '/home3/rvl/dataset/carla/v0.1'
        weight: 0.3
        camera: { ... }
      - root: '/home3/rvl/dataset/youtube_videos'
        weight: 0.7
        lut_train: '/home3/rvl/dataset/youtube_videos/lut_train_cs5_wp5.npz'
        lut_val:   '/home3/rvl/dataset/youtube_videos/lut_val_cs5_wp5.npz'
        camera: { ... }
"""

from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from data.carla_feat_dataset import CarlaFeatDataset
from data.mixture_sampler import WeightedMixtureSampler

from .dataset import FilteredFeatDataset


class FilteredMixtureDataModule(pl.LightningDataModule):
    """Like UrbanNavFeatMixtureDataModule, but with optional per-dataset LUTs."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers

    def _make_sub_cfg(self, entry):
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        data = cfg_dict["data"]
        root = entry["root"]
        data["root_dir"] = root
        data["pose_dir"] = entry.get("pose_subdir", "pose")
        data["feature_dir"] = entry.get(
            "feature_dir", f"{root}/{entry.get('feature_subdir', 'dino')}"
        )
        data["rgb_dir"] = entry.get(
            "rgb_dir", f"{root}/{entry.get('rgb_subdir', 'rgb')}"
        )
        if "camera" in entry:
            data["camera"] = entry["camera"]
        if "keep_list" in entry:
            data["keep_list"] = entry["keep_list"]
        for key in ("num_train", "num_val", "num_test"):
            if key in entry:
                data[key] = entry[key]
            else:
                data.setdefault(key, 0)
        return OmegaConf.create(cfg_dict)

    def _make_dataset(self, entry, sub_cfg, mode: str):
        lut_key = f"lut_{mode}"
        lut_path = entry.get(lut_key)
        if lut_path:
            db_path = entry.get("db_path")
            return FilteredFeatDataset(lut_path, sub_cfg, mode, db_path=db_path)
        return CarlaFeatDataset(sub_cfg, mode=mode)

    def setup(self, stage=None):
        mixture = OmegaConf.to_container(self.cfg.data.mixture, resolve=True)
        weights = [e["weight"] for e in mixture]

        if stage in ("fit", "validate") or stage is None:
            train_ds, val_ds = [], []
            for entry in mixture:
                sub_cfg = self._make_sub_cfg(entry)
                train_ds.append(self._make_dataset(entry, sub_cfg, "train"))
                val_ds.append(self._make_dataset(entry, sub_cfg, "val"))
            self.train_dataset = ConcatDataset(train_ds)
            self.val_dataset = ConcatDataset(val_ds)
            self.train_weights = weights

        if stage == "test" or stage is None:
            test_ds = []
            for entry in mixture:
                sub_cfg = self._make_sub_cfg(entry)
                test_ds.append(self._make_dataset(entry, sub_cfg, "test"))
            self.test_dataset = ConcatDataset(test_ds)

    def train_dataloader(self):
        sampler = WeightedMixtureSampler(
            dataset_lengths=[len(ds) for ds in self.train_dataset.datasets],
            weights=self.train_weights,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
