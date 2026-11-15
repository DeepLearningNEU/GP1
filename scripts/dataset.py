"""PyTorch dataset for the prepared COCO pairs: normalized image tensor + CLIP token ids."""
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CocoPairs(Dataset):
    def __init__(self, split_dir, tokenizer, size=512):
        self.root = Path(split_dir)
        self.rows = [json.loads(l) for l in open(self.root / "metadata.jsonl")]
        self.tok = tokenizer
        self.tf = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),                 # [0, 1]
            transforms.Normalize([0.5], [0.5]),    # -> [-1, 1], the range SD's VAE expects
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids = self.tok(r["text"], padding="max_length", max_length=self.tok.model_max_length,
                       truncation=True, return_tensors="pt").input_ids[0]
        return {"pixel_values": self.tf(Image.open(self.root / r["file_name"]).convert("RGB")),
                "input_ids": ids, "text": r["text"]}
