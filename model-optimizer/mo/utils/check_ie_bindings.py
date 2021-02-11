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

import os

from version import get_mo_version
from extract_release_version import extract_release_version


def try_to_import_ie():
    try:
        from openvino.inference_engine import IECore, get_version
        from openvino.offline_transformations import ApplyMOCTransformations

        ie_version = get_version()
        mo_version = get_mo_version()

        if mo_version not in ie_version:
            extracted_release_version = extract_release_version()
            is_custom_mo_version = extracted_release_version == (None, None)
            warning_message = "\n            ".join([
                "MO and IE versions do no match: MO: {}, IE: {}".format(mo_version, ie_version),
                "Some ModelOptimizer functionality may not work.",
                "Please consider to build InferenceEngine python from source or install OpenVINO using pip install openvino{}".format(
                    "" if is_custom_mo_version else "=={}.{}".format(*extracted_release_version))
            ])
            print("[ WARNING ] {}".format(warning_message))

        print("[ IMPORT ] Successfully imported InferenceEngine Python modules")
        return True
    except ImportError as e:
        print("[ IMPORT ] ImportError: {}".format(e))
        return False


if __name__ == "__main__":
    if not try_to_import_ie():
        exit(1)
