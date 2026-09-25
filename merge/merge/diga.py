#!/usr/bin/env python3
"""Fixed full-language merger: one-pass GS on T/R, TA with alpha=0.2 elsewhere."""

from diga_fixed import merge_matrix as diga_merge_matrix, merge_non_target_ta
from diga_model_io import build_parser as _build_parser, merge_models as _merge_models


def build_parser():
    return _build_parser(__doc__)


def merge_models(args):
    _merge_models(args, diga_merge_matrix, merge_non_target_ta)


def main():
    merge_models(build_parser().parse_args())


if __name__ == "__main__":
    main()
