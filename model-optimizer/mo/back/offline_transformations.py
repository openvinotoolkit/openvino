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

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--apply_low_latency", action="store_true")
    # parser.add_argument("--apply_pruning", action="store_true")
    parser.add_argument("--path_to_model")
    args = parser.parse_args()
    path_to_model = args.path_to_model

    try:
        from openvino.inference_engine import IECore # pylint: disable=import-error
        from openvino.offline_transformations import ApplyMOCTransformations, CheckAPI # pylint: disable=import-error
    except Exception as e:
        print("[ WARNING ] {}".format(e))
        exit(1)

    CheckAPI()

    ie = IECore()
    net = ie.read_network(model=path_to_model + "_tmp.xml", weights=path_to_model + "_tmp.bin")
    net.serialize(path_to_model + ".xml", path_to_model + ".bin")
