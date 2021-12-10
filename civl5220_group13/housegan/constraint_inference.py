import cv2
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
        default=1e-3,
        help="number of iterations",
    )
    parser.add_argument(
        "--mask-constraint",
        type=Path,
        help="constraint mask",
        default="toy_masks/0.txt",
    )
    parser.add_argument(
        "--general-constraint",
        type=parse_bool,
        default=True,
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=4,
    )


class Postprocess(nn.Module):
    def forward(self, masks):
        ret = torch.zeros_like(masks)
        for i, mi in enumerate(masks):
            for j, mij in enumerate(mi):
                mask = torch.zeros_like(mij)
                arr = mij.detach().cpu().numpy()
                arr = ((arr > 0) * 255).astype(np.uint8)
                _, _, stats, _ = cv2.connectedComponentsWithStats(arr)

                # -1 is the whole image, pick -2
                stats = sorted(stats, key=lambda l: l[-1])
                if len(stats) > 1:
                    w0, h0, dw, dh, _ = stats[-2]
                    h1 = h0 + dh
                    w1 = w0 + dw
                    mask[h0:h1, w0:w1] = 1

                    mij = mask * mij
                    ret[i][j] = mij

                    # plt.subplot(211)
                    # plt.imshow(arr)
                    # plt.subplot(212)
                    # plt.imshow(mij.detach().cpu().numpy())
                    # plt.show()

        return ret


class GeneralConstraint(nn.Module):
    def overlap_loss(self, masks):
        """
        Args:
            masks: (b k h w)
        """
        loss = 0
        ijs = list(combinations(range(masks.shape[1]), 2))
        for i, j in ijs:
            mi, mj = map(lambda m: m.clamp(min=0), masks[:, [i, j]].unbind(dim=1))
            # minimize overlap
            loss += (mi * mj).mean()
        loss /= len(ijs)
        return loss

    def forward(self, masks):
        loss = 0
        # loss += self.concentration_loss(masks)
        loss += self.overlap_loss(masks)
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

    for p in generator.parameters():
        # this may make bakcward faster
        p.requires_grad_(False)

    data = load_data(args.path)

    postprocess = Postprocess()

    constraints = nn.ModuleList([])

    if args.mask_constraint is not None:
        constraint = MaskConstraint(args.mask_constraint)
        cmask = constraint.mask.cpu().numpy()
        constraints.append(constraint)
        del constraint
    else:
        cmask = None

    constraints.to(args.device)

    if args.general_constraint:
        constraints.append(GeneralConstraint())

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

        snapshot = dict(
            edges=edges,
            nodes=nodes,
            cmask=cmask,
            masks=[],
            iters=[],
        )

        pbar = tqdm.trange(args.num_iters + 1)
        for j in pbar:
            optimizer.zero_grad()
            fake_masks = generator(z, onehot_nodes, edges_tensor)
            fake_masks = postprocess(fake_masks)

            loss = torch.zeros([], requires_grad=True, device=args.device)
            for constraint in constraints:
                loss = loss + constraint(fake_masks)

            loss.backward()

            if j % 100 == 0:
                # too fast is hard to read
                desc = f"loss: {loss.item():.3f}, grad: {z.grad.norm().item():.3f}"
                pbar.set_description(desc)

            optimizer.step()

            if j % args.dump_every == 0:
                snapshot["masks"].append(fake_masks.detach().cpu().numpy())
                snapshot["iters"].append(j)
                path = Path(f"snapshots/{i}.pkl")
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(snapshot, f)
