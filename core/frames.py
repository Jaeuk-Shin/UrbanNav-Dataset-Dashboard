"""Frame abstraction layer for video and JPG-directory inputs."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm


class FrameRef:
    """Reference to a single video frame (used in place of Path)."""
    __slots__ = ("video_path", "frame_idx", "stem")

    def __init__(self, video_path, frame_idx):
        self.video_path = video_path
        self.frame_idx = frame_idx
        self.stem = f"{frame_idx:06d}"


class SegmentReader:
    """Efficiently reads frames from a JPG directory or a video file.

    For JPG segments the VideoCapture is *None* and ``load`` just opens the
    file.  For video segments a single VideoCapture is kept open and seeked
    per frame, avoiding the overhead of opening/closing per frame.
    """

    def __init__(self, frames):
        self._cap = None
        if frames and isinstance(frames[0], FrameRef):
            self._cap = cv2.VideoCapture(str(frames[0].video_path))

    def load(self, ref):
        try:
            if self._cap is None:
                return Image.open(ref).convert("RGB")
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, ref.frame_idx)
            ret, bgr = self._cap.read()
            if not ret:
                tqdm.write(f"  WARNING: cannot read frame {ref.stem} from {ref.video_path}")
                return None
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            stem = ref.stem if hasattr(ref, "stem") else str(ref)
            tqdm.write(f"  WARNING: skipping frame {stem}: {e}")
            return None

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PreloadedReader:
    """Serves frames that were pre-loaded into memory by prefetch_segments."""

    def __init__(self, frames):
        self._iter = iter(frames)

    def load(self, _ref):
        return next(self._iter)

    def close(self):
        pass


def prefetch_segments(segments, prefetch_depth=2):
    """Yield (seg, paths, reader) with background prefetching.

    While the caller processes segment *i*, up to *prefetch_depth* subsequent
    segments are being decoded in background threads, hiding I/O latency.
    """

    def _preload(paths):
        reader = SegmentReader(paths)
        frames = [reader.load(p) for p in paths]
        reader.close()
        return frames

    items = list(segments.items())
    if not items:
        return

    with ThreadPoolExecutor(max_workers=prefetch_depth) as pool:
        q = deque()
        idx = 0
        while idx < min(prefetch_depth, len(items)):
            seg, paths = items[idx]
            q.append((seg, paths, pool.submit(_preload, paths)))
            idx += 1

        while q:
            seg, paths, fut = q.popleft()
            if idx < len(items):
                ns, np_ = items[idx]
                q.append((ns, np_, pool.submit(_preload, np_)))
                idx += 1
            yield seg, paths, PreloadedReader(fut.result())


def load_frame(ref):
    """One-shot frame loader (opens/closes video per call — use SegmentReader for batches)."""
    try:
        if isinstance(ref, Path):
            return Image.open(ref).convert("RGB")

        cap = cv2.VideoCapture(str(ref.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref.frame_idx)
        ret, bgr = cap.read()
        cap.release()
        if not ret:
            tqdm.write(f"  WARNING: cannot read frame {ref.stem}")
            return None
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    except Exception as e:
        stem = ref.stem if hasattr(ref, "stem") else str(ref)
        tqdm.write(f"  WARNING: skipping frame {stem}: {e}")
        return None
