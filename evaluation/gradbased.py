import pickle
import numpy as np
import pandas as pd

from civl5220_group13.housegan.metrics import ContourMetrics, QualityMetrics
from civl5220_group13.housegan.inference import mask_to_box

from pathlib import Path

contour_metric = ContourMetrics()
quality_metric = QualityMetrics()

with open("./contours/irregular.txt") as f:
    contour = np.array([list(line) for line in f.read().splitlines()])
    assert contour.shape == (32, 32)
    contour = np.where(contour == "#", 1, -1)

eval_list = [10, 12, 13, 14, 22, 27, 32, 36, 43, 7]

paths = list(Path("../snapshots/irregular").glob("*.pkl"))
paths = [p for p in paths if int(p.stem) in eval_list]
print(len(paths))

# assert len(paths) == 10

scores = []

for path in paths:
    with open(path, "rb") as f:
        data = pickle.load(f)
    nodes = data["nodes"]

    if len(data["masks"][-1]) != 10:
        continue

    # for baseline, we just choose the first 10 generation
    for masks in data["masks"][-1]:
        boxes = np.array([mask_to_box(mask) for mask in masks]) / 32
        score = dict()
        score["contour"] = contour_metric(boxes, contour)
        score["quality"] = quality_metric(boxes, nodes)
        scores.append(score)

df = pd.DataFrame(scores)

df = df.mean().apply(lambda s: f"{s:.3f}") + df.std().apply(lambda s: f" $\pm$ {s:.3f}")
print(df.T)
