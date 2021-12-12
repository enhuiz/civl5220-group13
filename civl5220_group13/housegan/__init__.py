from . import (
    visualize_dataset,
    extract_edges,
    inference,
    constraint_inference,
    animate,
    batched_inference,
)


def add_argument(parser):
    return [
        visualize_dataset,
        extract_edges,
        inference,
        constraint_inference,
        animate,
        batched_inference,
    ]
