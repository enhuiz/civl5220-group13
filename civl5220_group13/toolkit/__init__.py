import sys
from argparse_node import ArgumentParserNode

from . import show_sample


def add_argument(parser):
    return [show_sample]


def main():
    parser = ArgumentParserNode(sys.modules[__name__])
    args = parser.parse_args()
    parser.start(args)


if __name__ == "__main__":
    main()
