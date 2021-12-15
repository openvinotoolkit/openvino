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


def add_layouts(ov_function, argv: argparse.Namespace):
    from openvino.preprocess import PrePostProcessor  # pylint: disable=no-name-in-module,import-error
    from openvino.runtime import Layout  # pylint: disable=import-error,no-name-in-module

    prep = PrePostProcessor(ov_function)
    layout_values = argv.layout_values
    if '' in layout_values:
        if len(ov_function.inputs) == 1:
            layout_values = {
                list(ov_function.input().get_tensor().get_names())[0]: {
                    'source_layout': layout_values[''].get('source_layout'),
                    'target_layout': layout_values[''].get('target_layout')
                }
            }
        else:
            input_names = [list(ov_input.get_tensor().get_names())[0] for ov_input in ov_function.inputs]
            raise Error('Layout without name can be specified for models with only one input, '
                        'but provided model has {} inputs: \'{}\'. '
                        'Please specify explicitly input/output name for --layout option'
                        .format(len(input_names), input_names))

    set_layout_names = set(layout_values.keys())
    for idx, ov_input in enumerate(ov_function.inputs):
        found = set.intersection(set(ov_input.get_tensor().get_names()), set_layout_names)
        assert len(found) <= 1, 'More then one name point to the same node'
        if len(found) == 1:
            node_name = list(found)[0]
            found_layout = layout_values[node_name]
            if found_layout['source_layout']:
                prep.input(node_name).model().set_layout(Layout(found_layout['source_layout']))
            if found_layout['target_layout']:
                prep.input(node_name).tensor().set_layout(Layout(found_layout['target_layout']))

    for idx, ov_output in enumerate(ov_function.outputs):
        found = set.intersection(set(ov_output.get_tensor().get_names()), set_layout_names)
        assert len(found) <= 1, 'More then one name point to the same node'
        if len(found) == 1:
            node_name = list(found)[0]
            found_layout = layout_values[node_name]
            if found_layout['source_layout']:
                prep.output(node_name).model().set_layout(Layout(found_layout['source_layout']))
            if found_layout['target_layout']:
                prep.output(node_name).tensor().set_layout(Layout(found_layout['target_layout']))
    prep.build()


def apply_offline_transformations(input_model: str, argv: argparse.Namespace):
    # This variable is only needed by GenerateMappingFile transformation
    # to produce correct mapping
    extract_names = argv.framework in ['tf', 'mxnet', 'kaldi']

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

    add_layouts(func, argv)  # TODO: replace with preprocessing

    apply_user_transformations(func, parse_transform(argv.transform))
    apply_moc_transformations(func)

    if "compress_fp16" in argv and argv.compress_fp16:
        compress_model(func)

    serialize(func, str(input_model + ".xml").encode('utf-8'), (input_model + ".bin").encode('utf-8'))
    path_to_mapping = input_model + ".mapping"
    generate_mapping_file(func, path_to_mapping.encode('utf-8'), extract_names)
