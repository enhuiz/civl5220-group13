import torch
import torch.nn as nn

from itertools import combinations
from ..utils import ROOM_CLASS


def inversed(l):
    return [ROOM_CLASS.inverse[e] for e in l]


class FIoUCriterion(nn.Module):
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

    def forward(self, nodes, masks):
        """
        Args:
            nodes: (n)
            masks: (b n h w)
        """
        masks = ((masks + 1) / 2).clamp(min=0)  # [-1, 1] -> [0, 1]

        losses = []

        ijs = list(combinations(range(len(nodes)), 2))

        for i, j in ijs:
            assert i < j

            ni, nj = nodes[[i, j]]
            ti, tj = map(self.get_node_type, [ni, nj])
            tij = ti + tj

            iou_is_good = False

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

            # batched mask
            mi, mj = masks[:, [i, j]]

            # sum keep batch
            sumkb = lambda t: t.flatten(1).sum(dim=1)

            iou = sumkb(mi * mj) / sumkb(mi + mj - mi * mj)

            if iou_is_good:
                # minimize negative iou
                losses.append(-iou)
            else:
                losses.append(iou)

        loss = torch.stack(losses).mean()

        return loss


if __name__ == "__main__":
    metric = FIoUCriterion()
    score = metric(
        torch.tensor([1, 2, 3]),
        torch.randn(2, 3, 3, 3),
    )
    print(score)
