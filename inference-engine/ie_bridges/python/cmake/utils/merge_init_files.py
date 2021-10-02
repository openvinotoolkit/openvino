# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_file', type=str, required=True)
parser.add_argument('-o', '--output_file', type=str, required=True)

args = parser.parse_args()

mode = 'a' if os.path.exists(args.output_file) else 'w'
with open(args.input_file) as input_file, open(args.output_file, mode) as output_file:
    lines = input_file.readlines()
    count = 0
    copied_lines = ["\n"]
    for idx, line in enumerate(lines):
        if line[0] == '#' and idx <= 2:
            count += 1
        if idx > 2 and count == 3:
            copied_lines.append(line)
    output_file.writelines(copied_lines)
