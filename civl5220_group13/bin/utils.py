import json
from bidict import bidict


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def translate(value, source: bidict, target: bidict, unk):
    if source.inverse[value] in target:
        return target[source.inverse[value]]
    return unk
