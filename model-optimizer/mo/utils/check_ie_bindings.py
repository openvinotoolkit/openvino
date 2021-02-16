#!/usr/bin/env python3

"""
 Copyright (C) 2021 Intel Corporation

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

import re
import argparse

try:
    # needed by find_ie_version.py which call check_ie_bindings.py as python script
    import version
except ImportError:
    import mo.utils.version

from extract_release_version import extract_release_version


def import_core_modules(silent: bool, path_to_module: str):
    try:
        from openvino.inference_engine import IECore, get_version
        from openvino.offline_transformations import ApplyMOCTransformations, CheckAPI

        ie_version = str(get_version())
        mo_version = str(version.get_version())

        if not silent:
            print("\t- {}: \t{}".format("InferenceEngine found in", path_to_module))
            print("{}: \t{}".format("InferenceEngine version", ie_version))
            print("{}: \t    {}".format("Model Optimizer version", mo_version))

        # MO and IE version have a small difference in the beginning of version because
        # IE version also includes API version. For example:
        #   InferenceEngine version: 2.1.custom_HEAD_4c8eae0ee2d403f8f5ae15b2c9ad19cfa5a9e1f9
        #   Model Optimizer version:     custom_HEAD_4c8eae0ee2d403f8f5ae15b2c9ad19cfa5a9e1f9
        # So to match this versions we skip IE API version.
        if not re.match(r"^([0-9]+).([0-9]+).{}$".format(mo_version), ie_version):
            extracted_release_version = extract_release_version()
            is_custom_mo_version = extracted_release_version == (None, None)
            if not silent:
                print("[ WARNING ] MO and IE versions do no match.")
                print("[ WARNING ] Please consider to build InferenceEngine python from source or reinstall OpenVINO using pip install openvino{} {}".format(
                    "", "(may be incompatible with current ModelOptimizer version)" if is_custom_mo_version else "=={}.{}".format(*extracted_release_version), ""))

        return True
    except Exception as e:
        # Do not print a warning if module wasn't found or silent mode is on
        if "No module named 'openvino'" not in str(e) and not silent:
            print("[ WARNING ] {}".format(e))
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--path_to_module")
    args = parser.parse_args()

    if not import_core_modules(args.silent, args.path_to_module):
        exit(1)
