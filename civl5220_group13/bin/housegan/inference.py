import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from ...models.housegan.generator import Generator
from .utils import (
    draw_graph,
    draw_masks,
    bb_to_im_fid,
    mask_to_bb,
    read_houseganpp_graph,
    ROOM_CLASS,
)


def add_argument(parser):
    parser.add_argument(
        "sample",
        type=Path,
        help="json sample",
    )
    parser.add_argument(
        "ckpt",
        type=Path,
        help="pretrained checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device, cpu or cuda",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="dimensionality of the latent space",
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=4,
        help="number of variations",
    )


@torch.no_grad()
def start(args):
    generator = Generator()
    generator.load_state_dict(torch.load(args.ckpt, "cpu"))
    generator.to(args.device)

    # graph is a triple, masks (image), nodes and edges (graph)
    mks, rts, eds = read_houseganpp_graph(args.sample)
    nds = np.eye(len(ROOM_CLASS))[rts]

    # numpy to device tensor
    n2dt = lambda arr: torch.tensor(arr, device=args.device).float()
    mks, nds, eds = map(n2dt, [mks, nds, eds])

    graph_arr = draw_graph([rts, eds.cpu().numpy()])

    subplot = lambda n: plt.subplot(1, args.num_variations * 2 + 1, n)

    subplot(1)
    plt.imshow(graph_arr)

    for i in range(args.num_variations):
        z = torch.randn(mks.shape[0], args.latent_dim, device=args.device)

        gen_mks = generator(z, nds, eds)
        gen_bbs = np.array([np.array(mask_to_bb(mk)) for mk in gen_mks.detach().cpu()])
        gen_bbs = gen_bbs[np.newaxis, :, :] / 32.0

        subplot(2 * i + 2)
        fake_im_seg = draw_masks(gen_mks, rts)
        plt.imshow(fake_im_seg)

        subplot(2 * i + 3)
        fake_im_bb = bb_to_im_fid(gen_bbs, rts, im_size=256).convert("RGBA")
        plt.imshow(fake_im_bb)

    plt.show()
