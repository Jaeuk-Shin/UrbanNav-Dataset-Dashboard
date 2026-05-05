"""Base class for annotation pipeline stages."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from tqdm import tqdm

from pipeline.discovery import discover_segments, total_frames
from pipeline.frames import prefetch_segments

STAGES = {}


class BaseStage(ABC):
    """Abstract base for annotation pipeline stages.

    Subclasses set ``name`` and ``default_batch_size``, implement
    ``load_model`` and ``process_segment``, and optionally override
    ``should_skip``, ``add_arguments``, and ``on_complete``.

    Registration is automatic via ``__init_subclass__``.
    """

    name: str = ""
    default_batch_size: int = 32

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", ""):
            STAGES[cls.name] = cls

    @abstractmethod
    def load_model(self, device, args):
        """Load model(s) onto *device*. Store as instance attributes."""

    @abstractmethod
    def process_segment(self, seg_name, paths, reader, out_dir, args, pbar):
        """Process one segment. Must call ``pbar.update()`` for each frame."""

    def should_skip(self, out_dir, args):
        """Return ``True`` to skip this segment (called when not --overwrite)."""
        return False

    @classmethod
    def add_arguments(cls, parser):
        """Add stage-specific CLI arguments to *parser*."""

    def on_complete(self, out_root, args):
        """Called after all segments are processed."""
        print(f"Done ({self.name})")

    def run(self, args):
        """Template method: load -> discover -> process -> finish."""
        device = torch.device(args.device)
        args.batch_size = args.batch_size or self.default_batch_size
        self.load_model(device, args)

        data_root = Path(args.data_root)
        out_root = data_root / "annotations"

        segments = discover_segments(
            args.data_root, args.segments,
            names=getattr(args, "_segment_names", None),
            subsample=getattr(args, "subsample", 1) or 1,
            fps=getattr(args, "fps", None),
            input_format=args.input_format,
        )
        print(f"  {total_frames(segments)} frames across {len(segments)} segments")

        pbar = tqdm(total=total_frames(segments), desc=self.name)
        for seg, paths, reader in prefetch_segments(segments):
            out_dir = out_root / seg
            if not args.overwrite and self.should_skip(out_dir, args):
                pbar.update(len(paths))
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            self.process_segment(seg, paths, reader, out_dir, args, pbar)
        pbar.close()
        self.on_complete(out_root, args)
