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


def plot_all(nodes, edges, cmask, masks, iteration):
    plt.subplot(141)
    plot_graph(nodes, edges)

    plt.subplot(142)
    plot_masks([1], np.array([cmask]))

    plt.subplot(143)
    plot_masks(nodes, masks)
    plt.gca().text(0.4, 1.05, f"Iter: {iteration}", transform=plt.gca().transAxes)

    plt.subplot(144)
    plot_floorplan(nodes, masks)
    plt.gca().text(0.4, 1.05, f"Iter: {iteration}", transform=plt.gca().transAxes)


def main(args):
    path = args.path
    with open(path, "rb") as f:
        data = pickle.load(f)

    nodes = data["nodes"]
    edges = data["edges"]
    cmask = data["cmask"]
    iters = data["iters"]
    masks = list(zip(*data["masks"]))  # t b -> b t

    for i, mi in enumerate(tqdm.tqdm(masks)):
        fig = plt.figure(figsize=(20, 6))
        camera = Camera(fig)
        for mij, j in zip(mi, iters):
            plot_all(nodes, edges, cmask, mij, j)
            camera.snap()
        animation = camera.animate()
        animation.save(path.with_name(path.stem + f"-{i}.gif"))
        plt.close()
