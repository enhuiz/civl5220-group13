import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from celluloid import Camera


from .extract_edges import plot_graph
from .inference import plot_graph, plot_masks, plot_floorplan


def add_argument(parser):
    parser.add_argument("path", type=Path)


def plot_all(nodes, edges, cmask, masks):
    plt.subplot(141)
    plot_graph(nodes, edges)

    plt.subplot(142)
    plot_masks([1], np.array([cmask]))

    plt.subplot(143)
    plot_masks(nodes, masks)

    plt.subplot(144)
    plot_floorplan(nodes, masks)


def main(args):
    path = args.path
    with open(path, "rb") as f:
        data = pickle.load(f)

    nodes = data["nodes"]
    edges = data["edges"]
    cmask = data["cmask"]
    masks = list(zip(*data["masks"]))  # t b -> b t

    for i, mi in enumerate(tqdm.tqdm(masks)):
        fig = plt.figure(figsize=(20, 10))
        camera = Camera(fig)
        for mij in mi:
            plot_all(nodes, edges, cmask, mij)
            camera.snap()
        animation = camera.animate()
        animation.save(path.with_name(path.stem + f"-{i}.gif"))
