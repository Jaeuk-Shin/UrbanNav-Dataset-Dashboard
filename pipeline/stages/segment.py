"""Grounding DINO + SAM2 open-vocabulary segmentation stage."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pipeline.base import BaseStage

DEFAULT_CATEGORIES = (
    "person . car . truck . bus . van . motorcycle . bicycle . scooter . "
    "traffic light . traffic sign . stop sign . crosswalk . road . sidewalk . "
    "curb . lane marking . building . wall . fence . pole . tree . vegetation . "
    "grass . sky . bench . fire hydrant . bollard . barrier . manhole . staircase"
)


def _write_mask(mask_dir, idx, label_map, meta):
    """Write mask PNG and JSON metadata (safe to call from background thread)."""
    Image.fromarray(label_map).save(mask_dir / f"{idx}.png")
    (mask_dir / f"{idx}.json").write_text(json.dumps({"masks": meta}, indent=2))


def _write_empty_mask(mask_dir, idx, h, w):
    """Write empty mask files (safe to call from background thread)."""
    Image.fromarray(np.zeros((h, w), dtype=np.uint16)).save(
        mask_dir / f"{idx}.png"
    )
    (mask_dir / f"{idx}.json").write_text(json.dumps({"masks": {}}))


def _compose_label_map(pred_masks, boxes, labels, scores, h, w):
    """Build uint16 label map and metadata dict from predicted masks.

    Paints largest-area masks first so smaller objects appear on top.
    """
    if pred_masks.ndim == 4:
        pred_masks = pred_masks[:, 0]

    areas = [m.sum() for m in pred_masks]
    order = sorted(range(len(boxes)), key=lambda j: areas[j], reverse=True)

    label_map = np.zeros((h, w), dtype=np.uint16)
    meta = {}
    for mask_id, j in enumerate(order, start=1):
        binary = pred_masks[j]
        if binary.dtype != bool:
            binary = binary > 0.5
        label_map[binary] = mask_id
        meta[str(mask_id)] = {
            "label": labels[j],
            "score": round(float(scores[j]), 3),
            "bbox": [round(c, 1) for c in boxes[j].tolist()],
        }

    return label_map, meta


class SegmentStage(BaseStage):
    name = "segment"
    default_batch_size = 8

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--categories", default=None)
        parser.add_argument("--box-threshold", type=float, default=0.3)
        parser.add_argument("--use-caption-tags", action="store_true")

    def load_model(self, device, args):
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self._use_sam2 = True
        except ImportError:
            self._use_sam2 = False
            print("WARNING: sam2 package not installed -- falling back to SAM "
                  "ViT-Huge (much slower). Install with:  pip install SAM-2")

        # Grounding DINO (FP16)
        print("Loading Grounding DINO (base) [FP16] ...")
        gd_id = "IDEA-Research/grounding-dino-base"
        self.gd_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                gd_id, torch_dtype=torch.float16
            )
            .to(device)
            .eval()
        )
        self.gd_proc = AutoProcessor.from_pretrained(gd_id)

        # SAM
        if self._use_sam2:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            print("Loading SAM2 Hiera-Large ...")
            self.sam_predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-large", device=device,
            )
            self._sam2_batch = hasattr(self.sam_predictor, "set_image_batch")
        else:
            from transformers import SamModel, SamProcessor
            print("Loading SAM ViT-Huge ...")
            self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device).eval()
            self.sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")

        self.device = device
        self._io_pool = ThreadPoolExecutor(max_workers=4)

    def should_skip(self, out_dir, args):
        return (out_dir / "masks_done.json").exists()

    def process_segment(self, seg_name, paths, reader, out_dir, args, pbar):
        from torchvision.ops import nms

        categories = args.categories or DEFAULT_CATEGORIES
        box_thr = args.box_threshold
        use_tags = args.use_caption_tags
        bs = args.batch_size

        mask_dir = out_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        cap_tags = {}
        if use_tags:
            cf = out_dir / "captions.json"
            if cf.exists():
                cap_tags = json.loads(cf.read_text())

        # Pre-load all frames for the segment
        frames = []
        for p in paths:
            img = reader.load(p)
            if img is not None:
                frames.append((p, img))
            else:
                pbar.update(1)

        summary = {}
        io_futures = []

        # Process in batches
        for bi in range(0, len(frames), bs):
            batch = frames[bi : bi + bs]
            refs = [r for r, _ in batch]
            imgs = [img for _, img in batch]

            # Per-image prompts (may differ with --use-caption-tags)
            prompts = []
            for ref in refs:
                idx = ref.stem
                if use_tags and idx in cap_tags and cap_tags[idx].get("tags"):
                    prompts.append(" . ".join(cap_tags[idx]["tags"]))
                else:
                    prompts.append(categories)

            # Batched Grounding DINO
            target_sizes = [(img.height, img.width) for img in imgs]
            _t0 = torch.cuda.Event(enable_timing=True)
            _t1 = torch.cuda.Event(enable_timing=True)
            _t0.record()
            gd_in = self.gd_proc(
                images=imgs, text=prompts, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                gd_out = self.gd_model(**gd_in)
            _t1.record()
            torch.cuda.synchronize()
            _gd_ms = _t0.elapsed_time(_t1)

            all_dets = self.gd_proc.post_process_grounded_object_detection(
                gd_out,
                gd_in.input_ids,
                box_threshold=box_thr,
                text_threshold=box_thr,
                target_sizes=target_sizes,
            )

            # NMS per image; split into items with / without detections
            sam_items = []
            for ref, img, det in zip(refs, imgs, all_dets):
                idx = ref.stem
                w, h = img.size
                boxes = det["boxes"]
                labels = det["labels"]
                scores = det["scores"]

                if len(boxes) > 0:
                    keep = nms(boxes, scores, iou_threshold=0.5)
                    boxes = boxes[keep]
                    labels = [labels[k] for k in keep.tolist()]
                    scores = scores[keep]

                if len(boxes) == 0:
                    io_futures.append(
                        self._io_pool.submit(_write_empty_mask, mask_dir, idx, h, w)
                    )
                    summary[idx] = 0
                    pbar.update(1)
                else:
                    sam_items.append((ref, img, boxes, labels, scores, h, w))

            if not sam_items:
                continue

            # SAM dispatch — same per-item pipeline regardless of backend.
            _t2 = torch.cuda.Event(enable_timing=True)
            _t3 = torch.cuda.Event(enable_timing=True)
            _t2.record()
            if self._use_sam2 and self._sam2_batch and len(sam_items) > 1:
                pred_list = self._run_sam2_batch(sam_items)
            elif self._use_sam2:
                pred_list = self._run_sam2_per_image(sam_items)
            else:
                pred_list = self._run_sam1_fallback(sam_items)

            for item, pred_masks in zip(sam_items, pred_list):
                ref, _, boxes, labels, scores, h, w = item
                label_map, meta = _compose_label_map(
                    pred_masks, boxes, labels, scores, h, w
                )
                io_futures.append(self._io_pool.submit(
                    _write_mask, mask_dir, ref.stem, label_map, meta
                ))
                summary[ref.stem] = len(meta)
                pbar.update(1)

            _t3.record()
            torch.cuda.synchronize()
            _sam_ms = _t2.elapsed_time(_t3)
            n = len(sam_items)
            tqdm.write(
                f"  batch={len(batch)} gd={_gd_ms:.0f}ms "
                f"sam={_sam_ms:.0f}ms ({n} imgs, "
                f"{_sam_ms/max(n,1):.0f}ms/img)"
            )

        # Drain I/O futures for this segment
        for f in io_futures:
            f.result()

        (out_dir / "masks_done.json").write_text(json.dumps(summary))

    def _run_sam2_batch(self, sam_items):
        """Batched SAM2 image encoding + mask prediction."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_predictor.set_image_batch(
                [np.array(it[1]) for it in sam_items]
            )
            masks_list, _, _ = self.sam_predictor.predict_batch(
                box_batch=[it[2].cpu().numpy() for it in sam_items],
                multimask_output=False,
            )
        return list(masks_list)

    def _run_sam2_per_image(self, sam_items):
        """Per-image SAM2 (used when batched API is unavailable / single item)."""
        out = []
        for _, img, boxes, *_ in sam_items:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.sam_predictor.set_image(np.array(img))
                pred_masks, _, _ = self.sam_predictor.predict(
                    box=boxes.cpu().numpy(), multimask_output=False,
                )
            out.append(pred_masks)
        return out

    def _run_sam1_fallback(self, sam_items):
        """SAM ViT-Huge fallback — only used when ``sam2`` isn't installed."""
        out = []
        for _, img, boxes, *_ in sam_items:
            sam_in = self.sam_proc(
                img, input_boxes=[boxes.cpu().tolist()], return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                sam_out = self.sam_model(**sam_in)
            pred_masks_t = self.sam_proc.image_processor.post_process_masks(
                sam_out.pred_masks.cpu(),
                sam_in["original_sizes"].cpu(),
                sam_in["reshaped_input_sizes"].cpu(),
            )[0]
            best = sam_out.iou_scores.cpu()[0].argmax(dim=-1)
            pred_masks = np.stack([
                pred_masks_t[j, best[j].item()].numpy() > 0.5
                for j in range(len(boxes))
            ])
            out.append(pred_masks)
        return out

    def on_complete(self, out_root, args):
        self._io_pool.shutdown()
        print(f"Done -> {out_root}/*/masks/")
