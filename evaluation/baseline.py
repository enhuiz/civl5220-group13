import pickle
import numpy as np
import pandas as pd

from civl5220_group13.housegan.metrics import ContourMetrics, QualityMetrics
from pathlib import Path

contour_metric = ContourMetrics()
quality_metric = QualityMetrics()

with open("./contours/rectangle.txt") as f:
    contour = np.array([list(line) for line in f.read().splitlines()])
    assert contour.shape == (32, 32)
    contour = np.where(contour == "#", 1, -1)

paths = list(Path("./inferenced").glob("*.pkl"))
assert len(paths) == 10

scores = []

for path in paths:
    with open(path, "rb") as f:
        data = pickle.load(f)
    nodes = data["nodes"]

    # for baseline, we just choose the first 10 generation
    for boxes in data["fake_boxes"][:10]:
        score = dict()
        score["contour"] = contour_metric(boxes, contour)
        score["quality"] = quality_metric(boxes, nodes)
        scores.append(score)

df = pd.DataFrame(scores)

df = df.mean().apply(lambda s: f"{s:.3f}") + df.std().apply(lambda s: f" $\pm$ {s:.3f}")
print(df.T)
