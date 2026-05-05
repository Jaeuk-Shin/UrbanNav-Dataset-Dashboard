"""Florence-2 detailed captioning stage."""

import json

import torch

from pipeline.base import BaseStage


def _florence(model, processor, image, task, device, dtype=torch.float16, image_size=None):
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"].to(dtype),
            max_new_tokens=1024,
            num_beams=3,
        )
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        text, task=task, image_size=image_size or (image.width, image.height)
    )


class CaptionStage(BaseStage):
    name = "caption"
    default_batch_size = 1

    def load_model(self, device, args):
        from unittest.mock import patch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports

        # Florence-2's remote code lists flash_attn as an import even though
        # it's unused with the default attention implementation.
        def _florence_imports(filename) -> list[str]:
            imports = get_imports(filename)
            if str(filename).endswith("modeling_florence2.py"):
                imports = [i for i in imports if i != "flash_attn"]
            return imports

        print("Loading Florence-2-large ...")
        model_id = "microsoft/Florence-2-large"
        with patch("transformers.dynamic_module_utils.get_imports", _florence_imports):
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float16, trust_remote_code=True
                )
                .to(device)
                .eval()
            )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.device = device

    def should_skip(self, out_dir, args):
        return (out_dir / "captions.json").exists()

    def process_segment(self, seg_name, paths, reader, out_dir, args, pbar):
        caps = {}
        for p in paths:
            img = reader.load(p)
            if img is None:
                pbar.update(1)
                continue
            w, h = img.size
            idx = p.stem

            cap = _florence(self.model, self.processor, img, "<MORE_DETAILED_CAPTION>", self.device)
            od = _florence(self.model, self.processor, img, "<OD>", self.device, image_size=(w, h))
            tags = sorted(set(od.get("<OD>", {}).get("labels", [])))

            caps[idx] = {
                "caption": cap.get("<MORE_DETAILED_CAPTION>", ""),
                "tags": tags,
            }
            pbar.update(1)

        (out_dir / "captions.json").write_text(json.dumps(caps, indent=2))

    def on_complete(self, out_root, args):
        print(f"Done -> {out_root}/*/captions.json")
