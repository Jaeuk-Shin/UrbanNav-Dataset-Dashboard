"""Grounding DINO zero-shot crosswalk detection stage (bounding boxes only)."""

import json

import torch
from tqdm import tqdm

from stages.base import BaseStage

DEFAULT_PROMPT = "crosswalk"


class CrosswalkStage(BaseStage):
    name = "crosswalk"
    default_batch_size = 8

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--crosswalk-prompt", default=DEFAULT_PROMPT,
            help="Text prompt for Grounding DINO (default: crosswalk)")
        parser.add_argument(
            "--crosswalk-box-threshold", type=float, default=0.3,
            help="Box confidence threshold (default: 0.3)")
        parser.add_argument(
            "--crosswalk-nms-threshold", type=float, default=0.5,
            help="NMS IoU threshold (default: 0.5)")

    def load_model(self, device, args):
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        print("Loading Grounding DINO (base) [FP16] ...")
        gd_id = "IDEA-Research/grounding-dino-base"
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                gd_id, torch_dtype=torch.float16
            )
            .to(device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(gd_id)
        self.device = device
        self.prompt = args.crosswalk_prompt
        self.box_thr = args.crosswalk_box_threshold
        self.nms_thr = args.crosswalk_nms_threshold

        print(f"  Prompt: \"{self.prompt}\"  box_thr={self.box_thr}  nms_thr={self.nms_thr}")

    def should_skip(self, out_dir, args):
        return (out_dir / "crosswalks.json").exists()

    def process_segment(self, seg_name, paths, reader, out_dir, args, pbar):
        from torchvision.ops import nms

        bs = args.batch_size
        dets = {}

        for i in range(0, len(paths), bs):
            batch = paths[i : i + bs]
            loaded, valid_refs = [], []
            for p in batch:
                img = reader.load(p)
                if img is not None:
                    loaded.append(img)
                    valid_refs.append(p)
            if not loaded:
                pbar.update(len(batch))
                continue

            prompts = [self.prompt] * len(loaded)
            target_sizes = [(img.height, img.width) for img in loaded]

            gd_in = self.processor(
                images=loaded, text=prompts, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                gd_out = self.model(**gd_in)

            all_results = self.processor.post_process_grounded_object_detection(
                gd_out,
                gd_in.input_ids,
                box_threshold=self.box_thr,
                text_threshold=self.box_thr,
                target_sizes=target_sizes,
            )

            for ref, det in zip(valid_refs, all_results):
                boxes = det["boxes"]
                scores = det["scores"]
                labels = det["labels"]

                if len(boxes) > 0:
                    keep = nms(boxes, scores, iou_threshold=self.nms_thr)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = [labels[k] for k in keep.tolist()]

                entries = []
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    entries.append({
                        "bbox": [round(x1, 1), round(y1, 1),
                                 round(x2, 1), round(y2, 1)],
                        "confidence": round(float(score), 3),
                        "label": label,
                    })
                dets[ref.stem] = {"count": len(entries), "detections": entries}

            pbar.update(len(batch))

        (out_dir / "crosswalks.json").write_text(json.dumps(dets, indent=2))

    def on_complete(self, out_root, args):
        print(f"Done -> {out_root}/*/crosswalks.json")
