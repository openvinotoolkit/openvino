#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import os
import subprocess
import sys

from glob import glob

def find_deps(ov_package_path, deps_package_names):
    wheel_deps_list_to_find = deps_package_names.copy()
    wheel_deps_list_to_find.extend([d.replace('-', '_') for d in deps_package_names])
    wheels_dep_paths = []
    for d in wheel_deps_list_to_find:
        search_glob = os.path.join(ov_package_path, "**/" , "*" + d + "*.whl")
        for filename in glob(search_glob, recursive=True):
            wheels_dep_paths.append(filename)
    if len(wheels_dep_paths) != len(deps_package_names):
        raise RuntimeError(f"Cannot find all required dependencies {deps_package_names} in '*.whl' from OV package by path: {ov_package_path}")
    return wheels_dep_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("ov_package_path", help="Path to openVINO package")
    args = parser.parse_args()
    curr_dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(curr_dir_path, "inner_ov_package_requirements.txt"), "r") as inner_req_file:
        found_deps = find_deps(args.ov_package_path, inner_req_file.read().splitlines())
        generated_file_name = os.path.join(curr_dir_path, "generated_gathered_requirements_from_ov_package.txt")
        with open(generated_file_name, "w") as generated_desp_file:
            generated_desp_file.write("\n".join(found_deps))
        print(f"Wheels dependencies have been written into: {generated_file_name}")
