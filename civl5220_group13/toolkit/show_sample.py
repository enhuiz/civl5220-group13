import json
import matplotlib.pyplot as plt
import bbox_visualizer as bbv
import numpy as np
from PIL import Image


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def add_argument(parser):
    parser.add_argument("sample", type=load_json)
    parser.add_argument("--size", nargs=2, type=int, default=[256, 256])


def plot_boxes(boxes, size):
    img = Image.new("RGB", size)
    img = np.array(img)

    def rescale(box):
        box = np.array(box)
        box[::2] *= size[0] / 256
        box[1::2] *= size[1] / 256
        return box.astype(int)

    boxes = [rescale(box) for box in boxes]
    img = bbv.draw_multiple_rectangles(img, boxes, thickness=1)
    plt.imshow(img)


def start(args):
    sample = args.sample
    print("Keys and corresponding #elements:", {k: len(v) for k, v in sample.items()})
    plot_boxes(sample["boxes"], args.size)
    plt.savefig("tmp.png")
