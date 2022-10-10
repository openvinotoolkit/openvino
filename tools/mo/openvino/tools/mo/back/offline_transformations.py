# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import List

from openvino.tools.mo.front.extractor import create_params_with_custom_types
from openvino.tools.mo.utils.cli_parser import parse_transform
from openvino.tools.mo.utils.error import Error
from openvino.runtime import Model

def get_available_transformations():
    try:
        from openvino._offline_transformations import apply_low_latency_transformation # pylint: disable=import-error,no-name-in-module
        from openvino._offline_transformations import apply_make_stateful_transformation # pylint: disable=import-error,no-name-in-module
        from openvino._offline_transformations import apply_pruning_transformation # pylint: disable=import-error,no-name-in-module
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
    from openvino._offline_transformations import apply_moc_transformations  # pylint: disable=import-error,no-name-in-module
    apply_moc_transformations(func, False)


def apply_moc_legacy_transformations(func: object, params_with_custom_types: List[str]):
    from openvino._offline_transformations import apply_moc_legacy_transformations  # pylint: disable=import-error,no-name-in-module
    apply_moc_legacy_transformations(func, params_with_custom_types)


def compress_model(func: object):
    from openvino._offline_transformations import compress_model_transformation  # pylint: disable=import-error,no-name-in-module
    compress_model_transformation(func)

def apply_fused_names_cleanup(func: object):
    from openvino.offline_transformations import apply_fused_names_cleanup  # pylint: disable=import-error,no-name-in-module
    apply_fused_names_cleanup(func)


def apply_offline_transformations(func: Model, argv: argparse.Namespace):
    from openvino.tools.mo.back.preprocessing import apply_preprocessing  # pylint: disable=no-name-in-module,import-error

    # Apply preprocessing (mean/scale/reverse_channels/convert_layout/etc)
    apply_preprocessing(ov_function=func, argv=argv)

    apply_moc_transformations(func)

    params_with_custom_types = create_params_with_custom_types(argv.packed_user_shapes)
    apply_moc_legacy_transformations(func, params_with_custom_types)
    apply_user_transformations(func, parse_transform(argv.transform))

    if "compress_fp16" in argv and argv.compress_fp16:
        compress_model(func)

    apply_fused_names_cleanup(func)

    return func

