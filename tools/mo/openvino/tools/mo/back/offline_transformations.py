# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.cli_parser import parse_transform


def get_available_transformations():
    try:
        from openvino.offline_transformations_pybind import apply_low_latency_transformation, apply_make_stateful_transformation # pylint: disable=import-error,no-name-in-module
        return {
            'MakeStateful': apply_make_stateful_transformation,
            'LowLatency2': apply_low_latency_transformation,
        }
    except Exception as e:
        return {}


# net should be openvino.inference_engine.IENetwork type, but IE Engine is still optional dependency
def apply_user_transformations(func: object, transforms: list):
    available_transformations = get_available_transformations()

    for name, args in transforms:
        if name not in available_transformations.keys():
            raise Error("Transformation {} is not available.".format(name))

        available_transformations[name](func, **args)


def apply_moc_transformations(func: object):
    from openvino.offline_transformations_pybind import apply_moc_transformations  # pylint: disable=import-error,no-name-in-module
    apply_moc_transformations(func, False)

def compress_model(func: object):
    from openvino.offline_transformations_pybind import compress_model_transformation  # pylint: disable=import-error,no-name-in-module
    compress_model_transformation(func)

def apply_offline_transformations(input_model: str, framework: str, transforms: list, compress_fp16=False):
    # This variable is only needed by GenerateMappingFile transformation
    # to produce correct mapping
    extract_names = framework in ['tf', 'mxnet', 'kaldi']

    from openvino.offline_transformations_pybind import generate_mapping_file, serialize  # pylint: disable=import-error,no-name-in-module
    from openvino.frontend import FrontEndManager, FrontEnd  # pylint: disable=no-name-in-module,import-error

    fem = FrontEndManager()

    # We have to separate fe object lifetime from fem to
    # avoid segfault during object destruction. So fe must
    # be destructed before fem object explicitly.
    def read_model(path_to_xml):
        fe = fem.load_by_framework(framework="ir")
        function = fe.convert(fe.load(path_to_xml))
        return function

    func = read_model(input_model + "_tmp.xml")

    apply_user_transformations(func, transforms)
    apply_moc_transformations(func)

    if compress_fp16:
        compress_model(func)

    serialize(func, str(input_model + ".xml").encode('utf-8'), (input_model + ".bin").encode('utf-8'))
    path_to_mapping = input_model + ".mapping"
    generate_mapping_file(func, path_to_mapping.encode('utf-8'), extract_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model")
    parser.add_argument("--framework")
    parser.add_argument("--transform")
    parser.add_argument("--compress_fp16", action='store_true')
    args = parser.parse_args()

    apply_offline_transformations(args.input_model, args.framework, parse_transform(args.transform), args.compress_fp16)
