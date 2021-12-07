import numpy as np
from bidict import bidict


ROOM_CLASS = bidict(
    # from: https://github.com/ennauata/housegan/blob/master/utils.py
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
    # from: https://github.com/ennauata/housegan/blob/master/utils.py
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


def load_data(path, filter=True):
    """Load data
    Args:
        path: npy path
        filter: whether to remove bad data (from housegan author)
    Returns:
        data: [(nodes, boxes)]
    """
    data = np.load(path, allow_pickle=True)

    if filter:
        data = filter_graphs(data)

    for sample in data:
        nodes, boxes, _ = sample
        nodes, boxes = map(np.array, [nodes, boxes])
        # divided by 256 to normalize the coordinates
        boxes = boxes / 256.0
        yield (nodes, boxes)
