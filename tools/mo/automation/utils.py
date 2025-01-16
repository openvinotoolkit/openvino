# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec
import tarfile
from datetime import datetime
from shutil import copy, copytree, rmtree



class Automation:
    @staticmethod
    def parse_bom(bom_path):
        files = []
        for file in open(bom_path):
            files.append(file)
        return files

    @staticmethod
    def copy_files_from_bom(root_path, bom):
        target_dir = os.path.join(os.path.dirname(__file__), "ModelOptimizerForTensorflow")
        if os.path.exists(target_dir):
            rmtree(target_dir)
        os.makedirs(target_dir)
        for file in bom:
            src = os.path.join(root_path, file.strip('\n'))
            dst = os.path.join(target_dir, file.strip('\n'))
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            if os.path.isdir(src):
                copytree(src, dst)
            else:
                copy(src, dst)
        return target_dir

    @staticmethod
    def add_version_txt(dst_path, build_number):
        timestamp = datetime.now().strftime("%I:%M%p %B %d, %Y")
        with open(os.path.join(dst_path, "version.txt"), 'w') as f:
            f.write(timestamp + '\n')
            f.write(build_number + '\n')

    @staticmethod
    def make_tarfile(out_file_name, source_dir):
        archive_path = os.path.join(os.path.dirname(__file__), out_file_name)
        if os.path.exists(archive_path):
            os.remove(archive_path)
        with tarfile.open(out_file_name, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
