import numpy as np
from dataclasses import dataclass


@dataclass(eq=False)
class MetricBase:
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

    def draw_box(self, box):
        return self.draw_boxes([box])
