import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from einops import repeat
from pathlib import Path


class CIoUCriterion(nn.Module):
    def __init__(self, path):
        super().__init__()
        path = Path(path)
        if path.suffix in [".png", ".jpg", ".jpeg"]:
            mask = Image.open(path)
            mask = mask.resize((32, 32))
            mask = np.array(mask) / 255.0
            mask = torch.from_numpy(mask)
        else:
            with open(path, "r") as f:
                content = f.read().strip()
                content = content.replace("_", "0")
                content = content.replace("#", "1")
            lines = content.splitlines()
            lines = [list(map(int, line.strip())) for line in lines]
            mask = np.array(lines)
            assert mask.shape == (32, 32), f"but got {mask.shape}."
            mask = torch.from_numpy(mask)
        mask = torch.where(mask > 0.5, 1, 0)
        self.register_buffer("mask", mask)
        self.mask: torch.Tensor

    def forward(self, masks, **_):
        masks = ((masks + 1) / 2).clamp(min=0)  # [-1, 1] -> [0, 1]
        fake = masks.max(dim=1).values
        real = repeat(self.mask, "... -> b ...", b=len(masks))
        sumkb = lambda t: t.flatten(1).sum(dim=1)
        iou = sumkb(fake * real) / sumkb(fake + real - fake * real)
        loss = (1 - iou).mean()
        return loss


if __name__ == "__main__":
    metric = CIoUCriterion("toy_masks/0.txt")
    score = metric(
        torch.randn(2, 3, 32, 32),
    )
    print(score)
