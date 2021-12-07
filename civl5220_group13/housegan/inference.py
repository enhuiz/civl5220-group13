import cv2
import webcolors
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw

from .models.generator import Generator
from .utils import ROOM_CLASS, ID_COLOR, load_data
from .visualize_dataset import plot_floorplan
from .extract_edges import extract_edges, plot_graph


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
        "--num_variations",
        type=int,
        default=2,
        help="number of variations",
    )


def plot_masks(masks, nodes, size=(256, 256)):
    masks = masks.detach().cpu().numpy()

    # let me try to understand this
    canvas = Image.new("RGBA", size, (255, 255, 255, 0))

    for mask, node in zip(masks, nodes):
        # when greater than 0, the room occupies that pixel, it is white (i.e. 255)
        # otherwise, it is black (i.e. 0)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        mask = Image.fromarray(mask)

        # let's consider a so called region, i.e. a 32x32 canvas
        layer = Image.new("RGBA", (32, 32), (0, 0, 0, 0))  # like photoshop layer
        rgba = (*webcolors.name_to_rgb(ID_COLOR[node]), 32)
        draw = ImageDraw.Draw(layer)
        draw.bitmap((0, 0), mask.convert("L"), fill=rgba)

        # put layer onto canvas
        layer = layer.resize(size)
        canvas.paste(Image.alpha_composite(canvas, layer))

    for mask, node in zip(masks, nodes):
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
        _, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # a grayscale layer, for contours plotting
        grayscale_layer = np.zeros((*size, 3)).astype(np.uint8)
        cv2.drawContours(grayscale_layer, contours, -1, (255, 255, 255), 2)
        grayscale_layer = Image.fromarray(grayscale_layer).convert("L")

        # a rgb layer
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        rgba = (*webcolors.name_to_rgb(ID_COLOR[node]), 256)
        draw = ImageDraw.Draw(layer)
        draw.bitmap((0, 0), grayscale_layer, fill=rgba)

        canvas.paste(Image.alpha_composite(canvas, layer))

    plt.imshow(canvas)


def mask_to_box(mask):
    # get masks pixels
    indices = np.array(np.where(mask > 0))

    if indices.shape[-1] == 0:
        return [0, 0, 0, 0]

    # Compute BBs
    y0, x0 = np.min(indices, -1)
    y1, x1 = np.max(indices, -1)

    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, 255), min(x1, 255)

    return [x0, y0, x1 + 1, y1 + 1]


@torch.no_grad()
def main(args):
    generator = Generator()
    generator.load_state_dict(torch.load(args.ckpt, "cpu"))
    generator.to(args.device)

    data = load_data(args.path)

    for nodes, boxes in data:
        edges = extract_edges(boxes)

        plt.subplot(131)
        plot_graph(nodes, edges)

        # minus 1 as nodes are not starting from 0 but 1
        onehot_nodes = np.eye(len(ROOM_CLASS))[nodes - 1]

        # on device
        onehot_nodes = torch.from_numpy(onehot_nodes).to(args.device).float()
        edges = torch.from_numpy(edges).to(args.device).float()

        z = torch.randn(len(onehot_nodes), generator.latent_dim, device=args.device)

        # mask is the raw output of the model
        fake_masks = generator(z, onehot_nodes, edges)
        fake_masks = fake_masks.detach().cpu()  # (k 32 32), k is #nodes

        plt.subplot(132)
        plot_masks(fake_masks, nodes)

        # box is always rectangle
        fake_boxes = np.array([mask_to_box(mask) for mask in fake_masks])
        fake_boxes = fake_boxes / 32  # 32 for the model output size

        plt.subplot(133)
        plot_floorplan(nodes, fake_boxes, im_size=256)
        plt.show()
