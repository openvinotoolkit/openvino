# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import shutil
from pathlib import Path


def copy_images(input_dir: Path, output_dir: Path):
    """
    Copy images from doxygen xml folder to sphinx folder
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.svg', '*.gif', '*.PNG', '*.JPG', '*.JPEG', '*.SVG', '*.GIF')
    for extension in extensions:
        for file in input_dir.glob(extension):
            shutil.copy(file, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path, help='Path to the folder containing images.')
    parser.add_argument('output_dir', type=Path, help='Path to the output folder')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    copy_images(input_dir, output_dir)


if __name__ == '__main__':
    main()
