# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from mo.utils.error import Error
from mo.utils.cli_parser import parse_transform


def get_available_transformations():
    try:
        from openvino.offline_transformations import ApplyLowLatencyTransformation  # pylint: disable=import-error,no-name-in-module
        return {
            'LowLatency2': ApplyLowLatencyTransformation,
        }
    except Exception as e:
        return {}


def apply_offline_transformations(input_model: str, framework: str, transforms: list):
    # This variable is only needed by GenerateMappingFile transformation
    # to produce correct mapping
    extract_names = framework in ['tf', 'mxnet', 'kaldi']

    from openvino.inference_engine import read_network  # pylint: disable=import-error,no-name-in-module
    from openvino.offline_transformations import ApplyMOCTransformations, GenerateMappingFile  # pylint: disable=import-error,no-name-in-module

    net = read_network(input_model + "_tmp.xml", input_model + "_tmp.bin")

    available_transformations = get_available_transformations()

    for name, args in transforms:
        if name not in available_transformations.keys():
            raise Error("Transformation {} is not available.".format(name))

        available_transformations[name](net, **args)

    ApplyMOCTransformations(net, False)
    net.serialize(input_model + ".xml", input_model + ".bin")
    path_to_mapping = input_model + ".mapping"
    GenerateMappingFile(net, path_to_mapping.encode('utf-8'), extract_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model")
    parser.add_argument("--framework")
    parser.add_argument("--transform")
    args = parser.parse_args()

    apply_offline_transformations(args.input_model, args.framework, parse_transform(args.transform))