# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--apply_low_latency", action="store_true")
    # parser.add_argument("--apply_pruning", action="store_true")
    parser.add_argument("--path_to_model")
    args = parser.parse_args()
    path_to_model = args.path_to_model

    try:
        from openvino.inference_engine import IECore, read_network, read_network_without_extensions # pylint: disable=import-error
        from openvino.offline_transformations import ApplyMOCTransformations, GenerateMappingFile, CheckAPI # pylint: disable=import-error
    except Exception as e:
        print("[ WARNING ] {}".format(e))
        exit(1)

    CheckAPI()

    net = read_network_without_extensions(path_to_model + "_tmp.xml", path_to_model + "_tmp.bin")
    net.serialize(path_to_model + ".xml", path_to_model + ".bin")
    path_to_mapping = path_to_model + ".mapping"
    GenerateMappingFile(net, path_to_mapping.encode('utf-8'))

