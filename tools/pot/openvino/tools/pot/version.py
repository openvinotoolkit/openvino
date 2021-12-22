# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess


def generate_pot_version():
    """
    Function generates version like in cmake
    custom_{branch_name}_{commit_hash}
    """
    try:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        return "custom_{}_{}".format(branch_name, commit_hash)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown version"


def get_version():
    version = generate_pot_version()
    if version == "unknown version":
        version_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "version.txt")
        if os.path.isfile(version_txt):
            with open(version_txt) as f:
                version = f.readline().replace('\n', '')
    return version
