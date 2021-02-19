"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
import subprocess


def get_version_file_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "version.txt")


def generate_mo_version():
    """
    Function generates version like in cmake
    custom_{branch_name}_{commit_hash}
    """
    try:
        branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        return "custom_{}_{}".format(branch_name, commit_hash)
    except Exception as e:
        return "unknown version"


def get_version():
    version_txt = get_version_file_path()
    if not os.path.isfile(version_txt):
        return generate_mo_version()
    with open(version_txt) as f:
        version = f.readline().replace('\n', '')
    return version
