import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from .models.generator import Generator
from .utils import ROOM_CLASS, load_data
from .visualize_dataset import plot_floorplan


def add_argument(parser):
    parser.add_argument("path", type=Path, help="npy data path")
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="pretrained checkpoint",
        required=True,
    )
    parser.add_argument(
        "--rooms",
        choices=ROOM_CLASS,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--edges",
        type=lambda s: list(map(int, s.split("-"))),
        nargs="+",
        required=True,
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
        default=2,
        help="number of variations",
    )


@torch.no_grad()
def main(args):
    generator = Generator()
    generator.load_state_dict(torch.load(args.ckpt, "cpu"))
    generator.to(args.device)

    data = load_data(args.path)

    n2dt = lambda arr: torch.tensor(arr, device=args.device).float()

    for nodes, boxes in data:
        edges = extract_edges_from_boxes()

        onehot_nodes = np.eye(len(ROOM_CLASS))[nodes]

        # numpy to device tensor
        nodes, edges = map(n2dt, [nodes, edges])

        graph_arr = draw_graph([rts, edges.cpu().numpy()])

        subplot = lambda n: plt.subplot(1, args.num_variations * 2 + 1, n)

        subplot(1)
        plt.imshow(graph_arr)

        for i in range(args.num_variations):
            z = torch.randn(len(rts), args.latent_dim, device=args.device)

            gen_mks = generator(z, nodes, edges)
            gen_bbs = np.array(
                [np.array(mask_to_bb(mk)) for mk in gen_mks.detach().cpu()]
            )
            gen_bbs = gen_bbs[np.newaxis, :, :] / 32.0

            subplot(2 * i + 2)
            fake_im_seg = draw_masks(gen_mks, rts)
            plt.imshow(fake_im_seg)

            subplot(2 * i + 3)
            plot_floorplan(gen_bbs, rts, im_size=256)
            plt.show()

        plt.show()
