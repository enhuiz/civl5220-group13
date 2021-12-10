import pickle
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from itertools import combinations
from einops import repeat

from .models.batch_generator import Generator
from .utils import ROOM_CLASS, load_data
from .extract_edges import extract_edges


def parse_bool(s):
    assert s.lower() in ["true", "false"]
    return s.lower() == "true"


def add_argument(parser):
    parser.add_argument("path", type=Path, help="npy data path")
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="pretrained checkpoint",
        default="3rdparty/housegan/checkpoints/exp_demo_D_500000.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device, cpu or cuda",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=10000,
        help="number of iterations",
    )
    parser.add_argument(
        "--dump-every",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="number of iterations",
    )
    parser.add_argument(
        "--mask-constraint",
        type=Path,
        help="constraint mask",
        default="toy_mask.txt",
    )
    parser.add_argument(
        "--general-constraint",
        type=parse_bool,
        default=False,
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=32,
    )


class GeneralConstraint(nn.Module):
    def forward(self, masks):
        # TODO:
        loss = 0
        ijs = list(combinations(range(len(masks)), 2))
        for i, j in ijs:
            # minimize overlap
            loss += (masks[i] * masks[j]).clamp(min=0).mean()
        loss /= len(ijs)
        return loss


class MaskConstraint(nn.Module):
    def __init__(self, path):
        super().__init__()
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

        ones = torch.ones_like(mask).float()
        mask = torch.where(mask > 0.5, ones, ones.neg())

        self.register_buffer("mask", mask)
        self.mask: torch.Tensor

    def forward(self, masks):
        target = repeat(self.mask, "... -> b ...", b=len(masks))
        return F.mse_loss(masks.mean(dim=1), target)


def main(args):
    generator = Generator()
    generator.load_state_dict(torch.load(args.ckpt, "cpu"))
    generator.to(args.device)

    data = load_data(args.path)

    if args.mask_constraint is None:
        mask_constraint = None
    else:
        mask_constraint = MaskConstraint(args.mask_constraint)
        mask_constraint.to(args.device)

    if args.general_constraint:
        general_constraint = GeneralConstraint()
    else:
        general_constraint = None

    for i, (nodes, boxes) in enumerate(data):
        edges = extract_edges(boxes)

        # minus 1 as nodes are not starting from 0 but 1
        onehot_nodes = np.eye(len(ROOM_CLASS))[nodes - 1]

        # on device
        onehot_nodes = torch.from_numpy(onehot_nodes).to(args.device).float()
        edges_tensor = torch.from_numpy(edges).to(args.device).float()

        z = torch.randn(
            args.num_variations,
            len(onehot_nodes),
            generator.latent_dim,
            device=args.device,
            requires_grad=True,
        )

        optimizer = torch.optim.AdamW([z], lr=args.lr)
        pbar = tqdm.trange(args.num_iters)

        data = dict(
            edges=edges,
            nodes=nodes,
            cmask=mask_constraint.mask.cpu().numpy(),
            masks=[],
        )

        for j in pbar:
            optimizer.zero_grad()
            fake_masks = generator(z, onehot_nodes, edges_tensor)

            loss = torch.zeros([], requires_grad=True, device=args.device)
            if mask_constraint is not None:
                loss = loss + mask_constraint(fake_masks)
            if general_constraint is not None:
                loss = loss + general_constraint(fake_masks)

            loss.backward()
            pbar.set_description(f"loss: {loss.item():.3f}")
            optimizer.step()

            if j % args.dump_every == 0:
                data["masks"].append(fake_masks.detach().cpu().numpy())
                path = Path(f"snapshots/{i}.pkl")
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(data, f)
