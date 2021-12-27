# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.cli_parser import parse_transform


def get_available_transformations():
    try:
        from openvino.offline_transformations_pybind import apply_low_latency_transformation # pylint: disable=import-error,no-name-in-module
        from openvino.offline_transformations_pybind import apply_make_stateful_transformation # pylint: disable=import-error,no-name-in-module
        from openvino.offline_transformations_pybind import apply_pruning_transformation # pylint: disable=import-error,no-name-in-module
        return {
            'MakeStateful': apply_make_stateful_transformation,
            'LowLatency2': apply_low_latency_transformation,
            'Pruning': apply_pruning_transformation,
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


def apply_offline_transformations(input_model: str, argv: argparse.Namespace):
    # This variable is only needed by GenerateMappingFile transformation
    # to produce correct mapping
    extract_names = argv.framework in ['tf', 'mxnet', 'kaldi']

    from openvino.offline_transformations_pybind import generate_mapping_file, serialize  # pylint: disable=import-error,no-name-in-module
    from openvino.frontend import FrontEndManager, FrontEnd  # pylint: disable=no-name-in-module,import-error
    from openvino.tools.mo.back.preprocessing import apply_preprocessing  # pylint: disable=no-name-in-module,import-error

    fem = FrontEndManager()

    # We have to separate fe object lifetime from fem to
    # avoid segfault during object destruction. So fe must
    # be destructed before fem object explicitly.
    def read_model(path_to_xml):
        fe = fem.load_by_framework(framework="ir")
        function = fe.convert(fe.load(path_to_xml))
        return function

    func = read_model(input_model + "_tmp.xml")

    # TODO: use ngraph preprocessing (Mean/Scale/ReverseInputChannels) for legacy frontends
    reverse_input_channels = False
    if 'reverse_input_channels' in argv:
        reverse_input_channels = argv.reverse_input_channels
        argv.reverse_input_channels = False
    mean_scale_values = {}
    if 'mean_scale_values' in argv:
        mean_scale_values = argv.mean_scale_values
        argv.mean_scale_values = {}
    scale = None
    if 'scale' in argv:
        scale = argv.scale
        argv.scale = None

    # Apply preprocessing for layouts only
    apply_preprocessing(ov_function=func, argv=argv)

    if 'reverse_input_channels' in argv:
        argv.reverse_input_channels = reverse_input_channels
    if 'mean_scale_values' in argv:
        argv.mean_scale_values = mean_scale_values
    if 'scale' in argv:
        argv.scale = scale

    apply_user_transformations(func, parse_transform(argv.transform))
    apply_moc_transformations(func)

    if "compress_fp16" in argv and argv.compress_fp16:
        compress_model(func)

    serialize(func, str(input_model + ".xml").encode('utf-8'), (input_model + ".bin").encode('utf-8'))
    path_to_mapping = input_model + ".mapping"
    generate_mapping_file(func, path_to_mapping.encode('utf-8'), extract_names)
