import pickle
import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch
from pathlib import Path

from .models.batch_generator import Generator
from .utils import ROOM_CLASS, load_data
from .extract_edges import extract_edges
from .criterions import NLCCSCriterion, CIoUCriterion, FIoUCriterion


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
        "--contour-mask",
        type=Path,
        default="toy_masks/0.txt",
    )
    parser.add_argument(
        "--criterions",
        nargs="+",
        default=["nlccs", "fiou", "ciou"],
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=4,
    )


def main(args):
    generator = Generator()
    generator.load_state_dict(torch.load(args.ckpt, "cpu"))
    generator.to(args.device)

    for p in generator.parameters():
        # this may make bakcward faster
        p.requires_grad_(False)

    data = load_data(args.path)

    criterions = nn.ModuleList([])

    cmask = None

    for cname in args.criterions:
        if cname == "ciou":
            criterion = CIoUCriterion(args.contour_mask)
            cmask = criterion.mask.cpu().numpy()
        elif cname == "fiou":
            criterion = FIoUCriterion()
        elif cname == "nlccs":
            criterion = NLCCSCriterion()
        else:
            raise NotImplementedError()
        criterions.append(criterion)
        del criterion

    criterions.to(args.device)

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

            loss = torch.zeros([], requires_grad=True, device=args.device)
            for criterion in criterions:
                loss = loss + criterion(masks=fake_masks, nodes=nodes)

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
