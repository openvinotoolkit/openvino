# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import shutil

from pathlib import Path


def copy_directory_structure(input_dir, output_dir):
    """
    Recursively find files and process them accordingly
    """
    try:
        shutil.copytree(input_dir, output_dir)
    except FileExistsError:
        print(f"Destination directory already exists.")
    files = os.listdir(input_dir)
    for file in files:
        source_path = os.path.join(input_dir, file)
        destination_path = os.path.join(output_dir, file)
        if os.path.isdir(source_path):
            copy_directory_structure(source_path, destination_path)
        else:
            shutil.copy(source_path, destination_path)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--filetype', choices=['md', 'rst'], help='Type of processed files, allowed md or rst.')
    parser.add_argument('--input_dir', type=Path, help='Path to a folder containing files.')
    parser.add_argument('--output_dir', type=Path, help='Path to the output folder.')
    parser.add_argument('--exclude_dir', type=Path, action='append', default=[], help='Ignore a folder.')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_directory_structure(input_dir, output_dir)


if __name__ == '__main__':
    main()
