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


def get_mo_version():
    version_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "version.txt")
    if not os.path.isfile(version_txt):
        return "unknown version"
    with open(version_txt) as f:
        version = f.readline().replace('\n', '')
    return version


def try_to_import_ie():
    try:
        from openvino.inference_engine import IECore, get_version
        # print("[ IMPORT ]     Successfully Imported: IECore")

        from openvino.offline_transformations import ApplyMOCTransformations
        # print("[ IMPORT ]     Successfully Imported: ApplyMOCTransformations")

        ie_version = get_version()
        mo_version = get_mo_version()

        if mo_version not in ie_version:
            print("[ WARNING ] MO and IE versions do no match: MO: {}, IE: {}".format(mo_version, ie_version))

        print("[ IMPORT ] Successfully imported InferenceEngine Python modules")
        return True
    except ImportError as e:
        # print("[ IMPORT ]     ImportError: {}".format(e))
        return False


if __name__ == "__main__":
    if not try_to_import_ie():
        exit(1)