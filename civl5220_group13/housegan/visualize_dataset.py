"""
This script help you visualize the pre-processed data from HouseGAN.

pip install matplotlib Image numpy bidict
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

from .utils import ID_COLOR, ROOM_CLASS, load_data


def plot_floorplan(nodes, boxes, im_size=256):
    """This function draws bboxes and nodes on a canvas to produce an image.
    Args:
        boxes: (n 4), n rooms, 4 normalized coordinates i.e. between [0, 1]
        nodes: (n), n room types
        im_size: canvas size
    """
    nodes = np.array(nodes)
    boxes = np.array(boxes)
    areas = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes])
    inds = np.argsort(areas)[::-1]
    boxes = boxes[inds]
    nodes = nodes[inds]
    im = Image.new("RGB", (im_size, im_size), "white")
    dr = ImageDraw.Draw(im)
    for (x0, y0, x1, y1), nd in zip(boxes, nodes):
        if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
            continue
        else:
            color = ID_COLOR[nd]
            dr.rectangle(
                (x0 * im_size, y0 * im_size, x1 * im_size, y1 * im_size),
                width=3,
                outline="black",
                fill=color,
            )

    plt.imshow(im)
    patches = [Patch(color=ID_COLOR[k], label=ROOM_CLASS[k]) for k in ROOM_CLASS]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


def add_argument(parser):
    parser.add_argument("path")
    parser.add_argument("--no-filter", action="store_true")


def main(args):
    # i'm skipping
    # parser = argparse.ArgumentParser()
    # add_argument(parser)
    # args = parser.parse_args()
    # here because of the using of argparse-node package
    data = load_data(args.path, not args.no_filter)
    for nodes, boxes in data:
        plot_floorplan(nodes, boxes)
        plt.show()


# def artificial_example():
#     """
#     This is an artificial example for your reference (the one in the progress report)
#     You may just ignore this function as it is never called
#     """
#     nodes = [1, 6, 3, 4]
#     boxes = np.array(
#         [
#             # [12, 24, 49, 74],
#             [12, 100, 101, 216],
#             [77, 75, 101, 99],
#             [51, 24, 101, 99],
#             [12, 76, 40, 99.0],
#         ]
#     )
#     return nodes, boxes
