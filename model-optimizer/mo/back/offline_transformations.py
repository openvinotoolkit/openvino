# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from mo.utils.error import Error
from mo.utils.cli_parser import parse_transform


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

    from openvino.offline_transformations import GenerateMappingFile, Serialize  # pylint: disable=import-error,no-name-in-module
    from openvino.inference_engine import IENetwork  # pylint: disable=import-error,no-name-in-module
    from ngraph.frontend import FrontEndManager, FrontEnd  # pylint: disable=no-name-in-module,import-error
    from ngraph.impl import Function  # from ngraph.impl.Function import to_capsule

    fem = FrontEndManager()

    # We have to separate fe object lifetime from fem to
    # avoid segfault during object destruction. So fe must
    # be destructed before fem object explicitly.
    def read_network(path_to_xml):
        fe = fem.load_by_framework(framework="ir")
        f = fe.convert(fe.load(path_to_xml))
        return IENetwork(Function.to_capsule(f))

    net = read_network(input_model + "_tmp.xml")

    apply_user_transformations(net, transforms)
    apply_moc_transformations(net)
    Serialize(net, str(input_model + ".xml").encode('utf-8'), (input_model + ".bin").encode('utf-8'))
    path_to_mapping = input_model + ".mapping"
    GenerateMappingFile(net, path_to_mapping.encode('utf-8'), extract_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model")
    parser.add_argument("--framework")
    parser.add_argument("--transform")
    args = parser.parse_args()

    apply_offline_transformations(args.input_model, args.framework, parse_transform(args.transform))