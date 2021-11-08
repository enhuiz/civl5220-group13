import cv2
import webcolors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from PIL import Image, ImageDraw
from bidict import bidict

from ..utils import load_json, translate
from ..houseganpp.utils import ROOM_CLASS as HOUSEGANPP_ROOM_CLASS

ROOM_CLASS = bidict(
    {
        "living_room": 1,
        "kitchen": 2,
        "bedroom": 3,
        "bathroom": 4,
        "missing": 5,
        "closet": 6,
        "balcony": 7,
        "corridor": 8,
        "dining_room": 9,
        "laundry_room": 10,
    }
)

ID_COLOR = {
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


def pad_im(cr_im, final_size=256, bkg_color="white"):
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new("RGB", (new_size, new_size), "white")
    padded_im.paste(
        cr_im, ((new_size - cr_im.size[0]) // 2, (new_size - cr_im.size[1]) // 2)
    )
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im


def draw_graph(g_true):
    # build true graph
    G_true = nx.Graph()
    colors_H = []
    for k, label in enumerate(g_true[0]):
        _type = label + 1
        if _type >= 0:
            G_true.add_nodes_from([(k, {"label": _type})])
            colors_H.append(ID_COLOR[_type])
    for k, m, l in g_true[1]:
        if m > 0:
            G_true.add_edges_from([(k, l)], color="b", weight=4)

    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog="neato")

    edges = G_true.edges()
    colors = ["black"] * len(edges)
    weights = [4] * len(edges)

    nx.draw(
        G_true,
        pos,
        node_size=1000,
        node_color=colors_H,
        font_size=0,
        font_weight="bold",
        edges=edges,
        edge_color=colors,
        width=weights,
    )
    plt.tight_layout()

    fig = plt.gcf()
    fig.canvas.draw()
    rgb_arr = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    rgb_arr = rgb_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return rgb_arr


def draw_masks(masks, real_nodes):
    # Semitransparent background.
    bg_img = Image.new("RGBA", (256, 256), (255, 255, 255, 0))

    for m, nd in zip(masks, real_nodes):

        # draw region
        reg = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        dr_reg = ImageDraw.Draw(reg)
        m[m > 0] = 255
        m[m < 0] = 0
        m = m.detach().cpu().numpy()
        m = Image.fromarray(m)
        color = ID_COLOR[nd + 1]
        r, g, b = webcolors.name_to_rgb(color)
        dr_reg.bitmap((0, 0), m.convert("L"), fill=(r, g, b, 32))
        reg = reg.resize((256, 256))

        bg_img.paste(Image.alpha_composite(bg_img, reg))

    for m, nd in zip(masks, real_nodes):
        cnt = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        dr_cnt = ImageDraw.Draw(cnt)

        mask = np.zeros((256, 256, 3)).astype("uint8")
        m[m > 0] = 255
        m[m < 0] = 0
        m = m.detach().cpu().numpy()[:, :, np.newaxis].astype("uint8")
        m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_AREA)
        ret, thresh = cv2.threshold(m, 127, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            contours = [c for c in contours]
        color = ID_COLOR[nd + 1]
        r, g, b = webcolors.name_to_rgb(color)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), 2)

        mask = Image.fromarray(mask)
        dr_cnt.bitmap((0, 0), mask.convert("L"), fill=(r, g, b, 256))

        bg_img.paste(Image.alpha_composite(bg_img, cnt))

    return bg_img


def bb_to_im_fid(bbs_batch, nodes, im_size=299):
    """
    convert bboxes to image (fid???)
    solid rectangles
    """
    nodes = np.array(nodes)
    bbs = np.array(bbs_batch[0])
    areas = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in bbs])
    inds = np.argsort(areas)[::-1]
    bbs = bbs[inds]
    nodes = nodes[inds]
    im = Image.new("RGB", (im_size, im_size), "white")
    dr = ImageDraw.Draw(im)
    for (x0, y0, x1, y1), nd in zip(bbs, nodes):
        if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
            continue
        else:
            color = ID_COLOR[int(nd) + 1]
            dr.rectangle(
                (x0 * im_size, y0 * im_size, x1 * im_size, y1 * im_size),
                width=3,
                outline="black",
                fill=color,
            )
    return im


def mask_to_bb(mask):
    """
    The method just use the lower left and upper right pixels
    """

    # get masks pixels
    inds = np.array(np.where(mask > 0))

    if inds.shape[-1] == 0:
        return [0, 0, 0, 0]

    # Compute BBs
    y0, x0 = np.min(inds, -1)
    y1, x1 = np.max(inds, -1)

    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, 255), min(x1, 255)

    return [x0, y0, x1 + 1, y1 + 1]


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


def build_graph(bbs, types, threshold=0.03):
    """
    This function build a graph from only room types and bboxes
    Args:
        bbs, bounding boxes
        types: list of integers
    """
    # create edges -- make order
    triples = []

    nodes = types

    # encode connections
    for k, l in combinations(range(len(nodes)), 2):
        if is_adjacent(bbs[k], bbs[l], threshold):
            triples.append([k, 1, l])  # 1 for connects
        else:
            triples.append([k, -1, l])  # -1 for not

    return bbs, nodes, triples


def translate(value, source: bidict, target: bidict, unk):
    key = source.inverse[value]
    if key in target:
        return target[key]
    return unk


def translate_houseganpp_to_housegan(room_type):
    return translate(
        room_type,
        HOUSEGANPP_ROOM_CLASS,
        ROOM_CLASS,
        ROOM_CLASS["missing"],
    )


def read_houseganpp_graph(path):
    """
    This function is to read the HouseGAN++ style json sample (shown in README)
    """
    obj = load_json(path)

    types = obj["room_type"]
    bbs = np.array(obj["boxes"]) / 256

    # remove doors
    types, bbs = zip(
        *[
            (t, b)
            for t, b in zip(types, bbs)
            if "door" not in HOUSEGANPP_ROOM_CLASS.inverse[t]
        ]
    )

    types = list(map(translate_houseganpp_to_housegan, types))
    graph = build_graph(bbs, types)

    return graph
