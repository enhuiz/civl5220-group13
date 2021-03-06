import cv2
import numpy as np
import torch
import torch.nn as nn


class NLCCSCriterion(nn.Module):
    """
    Non-largest connected component suppression.
    """

    def forward(self, masks, **_):
        """
        Args:
            masks: (b k h w)
        """
        masks = ((masks + 1) / 2).clamp(min=0)  # [-1, 1] -> [0, 1]

        losses = []

        for mask in masks.flatten(0, 1).unbind(0):
            # penalty mask
            pmask = torch.ones_like(mask)

            arr = mask.detach().cpu().numpy()
            arr = ((arr > 0.5) * 255).astype(np.uint8)
            _, _, stats, _ = cv2.connectedComponentsWithStats(arr)

            # -1 is the whole image, pick -2
            stats = sorted(stats, key=lambda l: l[-1])
            if len(stats) > 1:
                w0, h0, dw, dh, _ = stats[-2]
                h1 = h0 + dh
                w1 = w0 + dw
                # dont penalize the largest connected component
                pmask[h0:h1, w0:w1] = 0
                # only penalty when there is more than 1 connected component
                losses.append((pmask * mask).mean())

        loss = torch.stack(losses).mean()

        return loss


if __name__ == "__main__":
    criterion = NLCCSCriterion()
    print(criterion(torch.randn(3, 3, 5, 5)))
