"""YOLOv8 object detection stage."""

import json

from stages.base import BaseStage


class DetectStage(BaseStage):
    name = "detect"
    default_batch_size = 32

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--detect-classes", default="person",
            help="Comma-separated COCO class names to detect (default: person)")
        parser.add_argument(
            "--detect-conf", type=float, default=0.25,
            help="Confidence threshold for detections (default: 0.25)")

    def load_model(self, device, args):
        from ultralytics import YOLO

        print("Loading YOLOv8x ...")
        self.model = YOLO("yolov8x.pt")
        self.device_str = args.device

        # Resolve class names to COCO IDs
        name_to_id = {v: k for k, v in self.model.names.items()}
        requested = [c.strip() for c in args.detect_classes.split(",")]
        self.class_ids = []
        for name in requested:
            if name not in name_to_id:
                available = ", ".join(sorted(name_to_id.keys()))
                raise ValueError(
                    f"Unknown class '{name}'. Available: {available}")
            self.class_ids.append(name_to_id[name])
        self.id_to_name = self.model.names
        self.conf = args.detect_conf

        print(f"  Classes: {requested} -> IDs {self.class_ids}")

    def should_skip(self, out_dir, args):
        return (out_dir / "detections.json").exists()

    def process_segment(self, seg_name, paths, reader, out_dir, args, pbar):
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
            results = self.model.predict(
                loaded,
                classes=self.class_ids,
                device=self.device_str,
                verbose=False,
                conf=self.conf,
            )
            for p, r in zip(valid_refs, results):
                boxes = []
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    boxes.append({
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "confidence": round(b.conf[0].item(), 3),
                        "class": self.id_to_name[int(b.cls[0].item())],
                    })
                dets[p.stem] = {"count": len(boxes), "detections": boxes}
            pbar.update(len(batch))

        (out_dir / "detections.json").write_text(json.dumps(dets, indent=2))

    def on_complete(self, out_root, args):
        print(f"Done -> {out_root}/*/detections.json")
