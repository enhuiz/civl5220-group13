from . import visualize_dataset, extract_edges, inference, constraint_inference, animate


def add_argument(parser):
    return [visualize_dataset, extract_edges, inference, constraint_inference, animate]
