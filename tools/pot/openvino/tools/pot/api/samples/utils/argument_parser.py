# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser


def get_common_argparser():
    parser = ArgumentParser(description='Post-training Compression Toolkit Sample',
                            allow_abbrev=False)
    parser.add_argument(
        '-m',
        '--model',
        help='Path to the xml file with model',
        required=True)
    parser.add_argument(
        '-w',
        '--weights',
        help='Path to the bin file with model weights',
        required=False)
    parser.add_argument(
        '-d',
        '--dataset',
        help='Path to the directory with data',
        required=True)

    return parser
