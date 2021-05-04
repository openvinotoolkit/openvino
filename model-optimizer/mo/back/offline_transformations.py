# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def apply_offline_transformations(input_model: str, framework: str):
    # This variable is only needed by GenerateMappingFile transformation
    # to produce correct mapping
    extract_names = True if framework in ['tf', 'mxnet', 'kaldi'] else False

    try:
        from openvino.inference_engine import IECore, read_network # pylint: disable=import-error
        from openvino.offline_transformations import ApplyMOCTransformations, GenerateMappingFile, CheckAPI # pylint: disable=import-error
    except Exception as e:
        print("[ WARNING ] {}".format(e))
        exit(1)

    CheckAPI()

    net = read_network(input_model + "_tmp.xml", input_model + "_tmp.bin")
    net.serialize(input_model + ".xml", input_model + ".bin")
    path_to_mapping = input_model + ".mapping"
    GenerateMappingFile(net, path_to_mapping.encode('utf-8'), extract_names)

    return 0