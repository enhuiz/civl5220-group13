import sys
from argparse_node import ArgumentParserNode

from . import housegan


def add_argument(parser):
    return [housegan]


def main():
    parser = ArgumentParserNode(sys.modules[__name__], starter="main")
    args = parser.parse_args()
    parser.start(args)


if __name__ == "__main__":
    main()
