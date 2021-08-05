# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_file', type=str, required=True)
parser.add_argument('--module_names', type=str, nargs='*', required=True)
parser.add_argument('--debug_postfix', type=str, required=True)

args = parser.parse_args()

_file_mod = str()

for module_name in args.module_names:
    with open(args.input_file) as _file:
        _file_mod = _file.read().replace(f"static const char __pyx_k_{module_name}[] = \"{module_name}\";",
                                        f"static const char __pyx_k_{module_name}[] = \"{module_name}{args.debug_postfix}\";")

if _file_mod:
    with open(args.input_file, 'w') as _file:
        _file.write(_file_mod)
