# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_file', type=str, required=True)
parser.add_argument('--match_string', type=str, required=True)
parser.add_argument('--replace_string', type=str, required=True)
parser.add_argument('-o', '--output_file', type=str, required=False)

args = parser.parse_args()

output_file = args.input_file if not args.output_file else args.output_file

if args.match_string != " " and args.match_string != args.replace_string:
    with open(args.input_file) as _file:
        _file_mod = _file.read().replace(args.match_string, args.replace_string)
    with open(output_file, 'w') as _file:
        _file.write(_file_mod)
