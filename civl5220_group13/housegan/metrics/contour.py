"""
Contour Metrics based on IOU, thanks to Changyang.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class ContourMetrics:
    im_size: int = 256

    def draw_boxes(self, boxes):
        im_size = self.im_size
        ret = np.zeros((im_size, im_size))
        for box in boxes:
            # all values should be within [0, 1]
            # note that Python support 0 <= n <= 1
            # no need to 0 <= n and n <= 1
            if all([0 <= n <= 1 for n in box]):
                scale = lambda n: int(im_size * n)
                x0, y0, x1, y1 = map(scale, box)
                ret[y0:y1, x0:x1] = 1
        return ret

    def __call__(self, boxes, real):
        """
        Args:
            boxes: (n 4)
            real: real contour map, (1? 32 32), > 0 means yes, otherwise means no
        """
        real = np.where(np.array(real) > 0, 1.0, 0.0)

        if real.ndim == 3:
            real = np.squeeze(real, axis=0)  # (1 32 32) -> (32 32)

        real = cv2.resize(real, (self.im_size,) * 2, interpolation=cv2.INTER_AREA)
        fake = self.draw_boxes(boxes)  # fake contour map

        real, fake = map(lambda x: x.astype(np.uint8), [real, fake])

        i = real & fake
        u = real | fake

        iou = i.sum() / u.sum()

        return iou


if __name__ == "__main__":
    metric = ContourMetrics()
    real = [
        [0, 0],
        [1, 1],
    ]
    boxes = [
        [0, 0, 1, 1],
    ]
    print(metric(boxes, real))
    boxes = [
        [0.0, 0.5, 1, 1],
    ]
    print(metric(boxes, real))
