import cv2
import numpy as np

from .base import MetricBase


class ContourMetrics(MetricBase):
    """
    intersection over union
    """

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
