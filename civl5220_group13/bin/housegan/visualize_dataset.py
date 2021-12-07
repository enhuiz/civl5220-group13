"""
This script help you visualize the pre-processed data from HouseGAN.

pip install matplotlib Image numpy bidict
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image, ImageDraw
from bidict import bidict

ROOM_CLASS = bidict(
    {
        1: "living_room",
        2: "kitchen",
        3: "bedroom",
        4: "bathroom",
        5: "missing",
        6: "closet",
        7: "balcony",
        8: "corridor",
        9: "dining_room",
        10: "laundry_room",
    }
)


ID_COLOR = bidict(
    {
        1: "brown",
        2: "magenta",
        3: "orange",
        4: "gray",
        5: "red",
        6: "blue",
        7: "cyan",
        8: "green",
        9: "salmon",
        10: "yellow",
    }
)


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


def filter_graphs(graphs, min_h=0.03, min_w=0.03):
    """
    Remove those bad samples,
    adapted from https://github.com/ennauata/housegan/blob/1ad2b75e5cc3c0373a7d5806790b6823c550bed0/floorplan_dataset_maps.py#L35
    """
    for g in graphs:
        # retrieve data
        rooms_type = g[0]
        rooms_bbs = g[1]

        # discard broken samples
        check_none = np.sum([bb is None for bb in rooms_bbs])
        check_node = np.sum([nd == 0 for nd in rooms_type])
        if (len(rooms_type) == 0) or (check_none > 0) or (check_node > 0):
            continue

        # filter small rooms
        tps_filtered = []
        bbs_filtered = []
        for n, bb in zip(rooms_type, rooms_bbs):
            h, w = (bb[2] - bb[0]), (bb[3] - bb[1])
            if h > min_h and w > min_w:
                tps_filtered.append(n)
                bbs_filtered.append(bb)

        # update graph
        yield tps_filtered, bbs_filtered, g[2]


def artificial_example():
    """
    This is an artificial example for you to reference
    """
    nodes = [1, 6, 3, 4]
    boxes = np.array(
        [
            # [12, 24, 49, 74],
            [12, 100, 101, 216],
            [77, 75, 101, 99],
            [51, 24, 101, 99],
            [12, 76, 40, 99.0],
        ]
    )
    return nodes, boxes


def add_argument(parser):
    parser.add_argument("path")
    parser.add_argument("--no-filter", action="store_true")


def main(args):
    # i'm skipping
    # parser = argparse.ArgumentParser()
    # add_argument(parser)
    # args = parser.parse_args()
    # here because of the argparse-node module

    data = np.load(args.path, allow_pickle=True)

    if not args.no_filter:
        data = filter_graphs(data)

    for sample in data:
        nodes, boxes, _ = sample
        nodes, boxes = map(np.array, [nodes, boxes])

        # divided by 256 to normalize the coordinates
        boxes = boxes / 256.0

        plot_floorplan(nodes, boxes)
        plt.show()
