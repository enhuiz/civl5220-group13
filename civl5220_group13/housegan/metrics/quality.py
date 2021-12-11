import numpy as np
from itertools import combinations

from ..utils import ROOM_CLASS
from .base import MetricBase


def inversed(l):
    return [ROOM_CLASS.inverse[e] for e in l]


class QualityMetrics(MetricBase):
    separate_list = inversed(
        [
            "living_room",
            "kitchen",
            "bedroom",
            "missing",
            "balcony",
            "corridor",
            "dining_room",
        ]
    )

    flexible_list = inversed(
        [
            "bathroom",
            "laundry_room",
        ]
    )

    affiliated_list = inversed(
        [
            "closet",
        ]
    )

    def get_node_type(self, node):
        if node in self.separate_list:
            return "s"
        if node in self.flexible_list:
            return "f"
        assert node in self.affiliated_list
        return "a"

    def __call__(self, nodes, boxes):
        """
        Args:
            nodes: (n)
            boxes: (n 4)
        """
        scores = []

        ijs = list(combinations(range(len(boxes)), 2))

        for i, j in ijs:
            assert i < j

            ni, nj = nodes[[i, j]]
            ti, tj = map(self.get_node_type, [ni, nj])
            tij = ti + tj

            iou_is_good = 0

            # should be 9 types
            if tij in ["ss", "aa"]:
                # iou is bad
                iou_is_good = False
            elif tij in ["sf", "fs", "ff"]:
                # just continue
                continue
            elif tij in ["as", "sa", "af", "fa"]:
                # iou is good
                iou_is_good = True

            bi, bj = boxes[[i, j]]

            # box2mask
            mi, mj = map(self.draw_box, [bi, bj])

            iou = (mi & mj).sum() / (mi | mj).sum()

            if iou_is_good:
                scores.append(iou)
            else:
                scores.append(1 - iou)

        return np.mean(scores)


if __name__ == "__main__":
    metric = QualityMetrics()
