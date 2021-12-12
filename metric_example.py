import numpy as np
from civl5220_group13.housegan.metrics import ContourMetrics, QualityMetrics

contour_metric = ContourMetrics()
quality_metric = QualityMetrics()

boxes = np.random.randn(10, 4)
real = np.random.randn(32, 32)
nodes = np.arange(1, 11)

print("contour", contour_metric(boxes, real))
print(ContourMetrics.__call__.__doc__)

print("quality", quality_metric(boxes, nodes))
print(QualityMetrics.__call__.__doc__)
