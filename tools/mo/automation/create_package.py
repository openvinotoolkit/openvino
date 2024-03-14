# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from shutil import rmtree

from utils import Automation

parser = argparse.ArgumentParser()
parser.add_argument("--build_number", type=str, help="Build number to be added to package version", default="0", )
args = parser.parse_args()

auto = Automation()
base_dir = os.path.dirname(__file__)
bom_path = os.path.join(base_dir, "package_BOM.txt")
bom = auto.parse_bom(bom_path=bom_path)
dir_to_tar = auto.copy_files_from_bom(root_path=os.path.join(os.path.dirname(__file__), ".."), bom=bom)
auto.add_version_txt(dst_path=dir_to_tar, build_number=args.build_number)

auto.make_tarfile(out_file_name="mo_for_tf_{0}.tar.gz".format(args.build_number), source_dir=dir_to_tar)
rmtree(dir_to_tar)
