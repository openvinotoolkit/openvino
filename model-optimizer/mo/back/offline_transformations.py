# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model")
    parser.add_argument("--framework")
    args = parser.parse_args()
    path_to_model = args.input_model

    # This variable is only needed by GenerateMappingFile transformation
    # to produce correct mapping
    extract_names = True if args.framework in ['tf', 'mxnet', 'kaldi'] else False

    try:
        from openvino.inference_engine import IECore, read_network # pylint: disable=import-error
        from openvino.offline_transformations import ApplyMOCTransformations, GenerateMappingFile, CheckAPI # pylint: disable=import-error
    except Exception as e:
        print("[ WARNING ] {}".format(e))
        exit(1)

    CheckAPI()

    net = read_network(path_to_model + "_tmp.xml", path_to_model + "_tmp.bin")
    net.serialize(path_to_model + ".xml", path_to_model + ".bin")
    path_to_mapping = path_to_model + ".mapping"
    GenerateMappingFile(net, path_to_mapping.encode('utf-8'), extract_names)

