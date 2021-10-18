# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.cli_parser import parse_transform


def get_available_transformations():
    try:
        from openvino.offline_transformations import ApplyLowLatencyTransformation, ApplyMakeStatefulTransformation # pylint: disable=import-error,no-name-in-module
        return {
            'MakeStateful': ApplyMakeStatefulTransformation,
            'LowLatency2': ApplyLowLatencyTransformation,
        }
    except Exception as e:
        return {}


# net should be openvino.inference_engine.IENetwork type, but IE Engine is still optional dependency
def apply_user_transformations(net: object, transforms: list):
    available_transformations = get_available_transformations()

    for name, args in transforms:
        if name not in available_transformations.keys():
            raise Error("Transformation {} is not available.".format(name))

        available_transformations[name](net, **args)


def apply_moc_transformations(net: object):
    from openvino.offline_transformations import ApplyMOCTransformations  # pylint: disable=import-error,no-name-in-module
    ApplyMOCTransformations(net, False)


def apply_offline_transformations(input_model: str, framework: str, transforms: list):
    # This variable is only needed by GenerateMappingFile transformation
    # to produce correct mapping
    extract_names = framework in ['tf', 'mxnet', 'kaldi']

    from openvino.inference_engine import read_network  # pylint: disable=import-error,no-name-in-module
    from openvino.offline_transformations import GenerateMappingFile  # pylint: disable=import-error,no-name-in-module

    net = read_network(input_model + "_tmp.xml", input_model + "_tmp.bin")
    apply_user_transformations(net, transforms)
    apply_moc_transformations(net)
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