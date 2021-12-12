"""
Quality metrics, thanks to Kai.
"""

import numpy as np
from itertools import combinations
from ..utils import ROOM_CLASS


def inversed(l):
    return [ROOM_CLASS.inverse[e] for e in l]


class QualityMetrics:
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

    def bbox_cr(self, bbox1, bbox2):
        cx1, cy1, cx2, cy2 = bbox1
        gx1, gy1, gx2, gy2 = bbox2

        carea = (cx2 - cx1) * (cy2 - cy1)
        garea = (gx2 - gx1) * (gy2 - gy1)

        x1, y1, x2, y2 = max(cx1, gx1), max(cy1, gy1), min(cx2, gx2), min(cy2, gy2)
        w, h = max(0, x2 - x1), max(0, y2 - y1)

        area = w * h
        cr = area / (min(carea, garea) + 1e-7)

        return cr

    def get_node_type(self, node):
        if node in self.separate_list:
            return "s"
        if node in self.flexible_list:
            return "f"
        assert node in self.affiliated_list
        return "a"

    def __call__(self, boxes, nodes):
        """
        Args:
            boxes: (n 4)
            nodes: (n)
        """
        scores = []

        ijs = list(combinations(range(len(boxes)), 2))

        for i, j in ijs:
            assert i < j

            ni, nj = nodes[[i, j]]
            ti, tj = map(self.get_node_type, [ni, nj])
            tij = "".join(sorted(ti + tj))

            beta = 0

            # should be 9 types
            if tij in ["ss", "aa"]:
                # cr is bad
                beta = 0
            elif tij in ["fs", "ff"]:
                # just continue
                continue
            elif tij in ["as", "af"]:
                # cr is good
                beta = 1
            else:
                raise ValueError(tij)

            bi, bj = boxes[[i, j]]
            cr = self.bbox_cr(bi, bj)

            scores.append(np.abs(beta - cr))

        return 1 - np.mean(scores)


if __name__ == "__main__":
    metric = QualityMetrics()
    score = metric(
        np.array(
            [
                [0, 0, 1, 1],
                [0.5, 0.2, 1, 1],
                [0.5, 0.2, 1, 1],
            ]
        ),
        np.random.randint(1, 10, 3),
    )
    print(score)
