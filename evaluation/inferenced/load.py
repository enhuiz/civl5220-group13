import numpy as np
import pickle
import matplotlib.pyplot as plt

from civl5220_group13.housegan.inference import plot_floorplan_impl, plot_graph

# ids for you to make loading easier
eval_list = [10, 12, 13, 14, 22, 27, 32, 36, 43, 7]

with open("./7.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
plt.subplot(141)
plot_graph(data["nodes"], data["edges"])
plt.subplot(142)
plot_floorplan_impl(data["nodes"], data["real_boxes"])
plt.subplot(143)
print(data["fake_boxes"][0])
exit()
plot_floorplan_impl(data["nodes"], data["fake_boxes"][0])
plt.subplot(144)
plot_floorplan_impl(data["nodes"], data["fake_boxes"][4999])
plt.show()
