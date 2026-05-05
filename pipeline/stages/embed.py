"""SigLIP embedding stage."""

import json

import numpy as np
import torch
from tqdm import tqdm

from pipeline.base import BaseStage


class EmbedStage(BaseStage):
    name = "embed"
    default_batch_size = 64

    def load_model(self, device, args):
        import open_clip

        print("Loading SigLIP ViT-SO400M-14-SigLIP-384 ...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP-384", pretrained="webli"
        )
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.device = device

    def should_skip(self, out_dir, args):
        return (out_dir / "embeddings.npy").exists()

    def process_segment(self, seg_name, paths, reader, out_dir, args, pbar):
        bs = args.batch_size
        embs = []
        for i in range(0, len(paths), bs):
            batch = paths[i : i + bs]
            loaded, valid = [], []
            for p in batch:
                img = reader.load(p)
                if img is not None:
                    loaded.append(self.preprocess(img))
                    valid.append(True)
                else:
                    valid.append(False)
            if not loaded:
                pbar.update(len(batch))
                continue
            imgs = torch.stack(loaded).to(self.device)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                e = self.model.encode_image(imgs)
                e = e / e.norm(dim=-1, keepdim=True)
            e_np = e.cpu().half().numpy()
            full = np.zeros((len(batch), e_np.shape[1]), dtype=np.float16)
            full[np.array(valid)] = e_np
            embs.append(full)
            pbar.update(len(batch))

        if embs:
            np.save(out_dir / "embeddings.npy", np.concatenate(embs))
        else:
            tqdm.write(f"  WARNING: no readable frames in {seg_name}, skipping")
        return

    def on_complete(self, out_root, args):
        meta = {"model": "ViT-SO400M-14-SigLIP-384", "pretrained": "webli"}
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "embedding_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Done -> {out_root}/*/embeddings.npy")
        return