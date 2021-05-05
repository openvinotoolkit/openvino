# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from mo.utils.error import Error


def get_available_transformations():
    try:
        from openvino.offline_transformations import ApplyLowLatencyTransformation # pylint: disable=import-error
        return {
            'LowLatency': ApplyLowLatencyTransformation,
        }
    except Exception as e:
        return {}


def apply_offline_transformations(input_model: str, framework: str, transforms: list):
    # This variable is only needed by GenerateMappingFile transformation
    # to produce correct mapping
    extract_names = True if framework in ['tf', 'mxnet', 'kaldi'] else False

    try:
        from openvino.inference_engine import read_network # pylint: disable=import-error
        from openvino.offline_transformations import ApplyMOCTransformations, GenerateMappingFile # pylint: disable=import-error
    except Exception as e:
        print("[ WARNING ] {}".format(e))
        return 1

    net = read_network(input_model + "_tmp.xml", input_model + "_tmp.bin")

    available_transformations = get_available_transformations()

    for name, args in transforms:
        if name not in available_transformations.keys():
            raise Error("Transformation {} is not available.".format(name))

        print("[ INFO ] Applying {} with {} args".format(name, args))
        available_transformations[name](net, **args)

    net.serialize(input_model + ".xml", input_model + ".bin")
    path_to_mapping = input_model + ".mapping"
    GenerateMappingFile(net, path_to_mapping.encode('utf-8'), extract_names)

    return 0