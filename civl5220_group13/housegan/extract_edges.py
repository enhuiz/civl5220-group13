import tempfile
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from pygraphviz import AGraph

from .utils import ROOM_CLASS, load_data, ID_COLOR


def is_adjacent(box_a, box_b, threshold=0.03):
    x0, y0, x1, y1 = box_a
    x2, y2, x3, y3 = box_b

    h1, h2 = x1 - x0, x3 - x2
    w1, w2 = y1 - y0, y3 - y2

    xc1, xc2 = (x0 + x1) / 2.0, (x2 + x3) / 2.0
    yc1, yc2 = (y0 + y1) / 2.0, (y2 + y3) / 2.0

    delta_x = np.abs(xc2 - xc1) - (h1 + h2) / 2.0
    delta_y = np.abs(yc2 - yc1) - (w1 + w2) / 2.0

    delta = max(delta_x, delta_y)

    return delta < threshold


def extract_edges(boxes):
    edges = []
    for i, j in combinations(range(len(boxes)), 2):
        edges.append([i, 1 if is_adjacent(boxes[i], boxes[j]) else -1, j])
    return edges


def plot_graph(nodes, edges, show_name=True):
    """
    Args:
        nodes: list of intergers, starting from 1 (not 0).
        edges: extracted edges
    """
    graph = AGraph(strict=False, directed=False)

    for i, node in enumerate(nodes):
        color = ID_COLOR[node]
        name = str(i) + "." + ROOM_CLASS[node]
        graph.add_node(i, label=name if show_name else "", color=color)

    for i, p, j in edges:
        if p > 0:
            graph.add_edge(i, j, color="black", penwidth="4")

    graph.node_attr["style"] = "filled"
    graph.layout(prog="dot")

    file = tempfile.NamedTemporaryFile()
    graph.draw(file.name, "png")
    plt.imshow(plt.imread(file.name))

    # this also deletes the temp file from disk
    del file


def add_argument(parser):
    parser.add_argument("path")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--show-name", action="store_true")


def main(args):
    # i'm skipping
    # parser = argparse.ArgumentParser()
    # add_argument(parser)
    # args = parser.parse_args()
    # here because of the argparse-node module
    data = load_data(args.path, not args.no_filter)

    for nodes, boxes in data:
        edges = extract_edges(boxes)
        plot_graph(nodes, edges, args.show_name)
        plt.show()
