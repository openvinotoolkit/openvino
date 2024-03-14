# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from openvino.runtime import Model  # pylint: disable=no-name-in-module,import-error
from openvino.tools.mo.utils.cli_parser import parse_transform
from openvino.tools.mo.back.preprocessing import apply_preprocessing


def moc_emit_ir(ngraph_function: Model, argv: argparse.Namespace):

    # Apply preprocessing (mean/scale/reverse_channels/convert_layout/etc)
    apply_preprocessing(ov_function=ngraph_function, argv=argv)

    # Apply transformations
    from openvino.tools.mo.back.offline_transformations import apply_user_transformations, \
        apply_moc_legacy_transformations, apply_fused_names_cleanup

    from openvino._offline_transformations import apply_moc_transformations  # pylint: disable=import-error,no-name-in-module
    apply_moc_transformations(ngraph_function, cf=argv.static_shape, smart_reshape=True)

    from openvino._offline_transformations import compress_quantize_weights_transformation # pylint: disable=no-name-in-module,import-error
    compress_quantize_weights_transformation(ngraph_function)

    if argv.framework == "onnx":
        # set OldApi map in IR to be executed via OV API 1.x and for parity with legacy MO
        params_with_custom_types = [] if argv.placeholder_data_types is None \
            else list(argv.placeholder_data_types.keys())
        apply_moc_legacy_transformations(ngraph_function, params_with_custom_types)

    apply_user_transformations(ngraph_function, parse_transform(argv.transform))

    if argv.compress_to_fp16:
        from openvino.tools.mo.back.offline_transformations import compress_model
        compress_model(ngraph_function)

    apply_fused_names_cleanup(ngraph_function)

    del argv.feManager
    return ngraph_function
