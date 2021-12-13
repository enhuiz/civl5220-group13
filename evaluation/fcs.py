import pickle
import numpy as np
import pandas as pd

from civl5220_group13.housegan.metrics import ContourMetrics, QualityMetrics
from pathlib import Path

from fcs_result import rectangle, polygon

contour_metric = ContourMetrics()
quality_metric = QualityMetrics()


with open("./contours/rectangle.txt") as f:
    contour = np.array([list(line) for line in f.read().splitlines()])
    assert contour.shape == (32, 32)
    contour = np.where(contour == "#", 1, -1)

eval_list = [10, 12, 13, 14, 22, 27, 32, 36, 43, 7]
paths = list(Path("./inferenced").glob("*.pkl"))
assert len(paths) == 10

selections = np.load("fqs/result.npy")

scores = []

for i, path in enumerate(paths):
    selection = rectangle[str(int(path.stem))]
    with open(path, "rb") as f:
        data = pickle.load(f)
    nodes = data["nodes"]

    # for baseline, we just choose the first 10 generation
    for boxes in data["fake_boxes"][selection]:
        score = dict()
        score["contour"] = contour_metric(boxes, contour)
        score["quality"] = quality_metric(boxes, nodes)
        scores.append(score)

df = pd.DataFrame(scores)

df = df.mean().apply(lambda s: f"{s:.3f}") + df.std().apply(lambda s: f"±{s:.3f}")

print(" & ".join(df.tolist()))
