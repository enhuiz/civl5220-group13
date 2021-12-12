import numpy as np
import pickle

# ids for you to make loading easier
eval_list = [10, 12, 13, 14, 22, 27, 32, 36, 43, 7]

with open("./0.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
