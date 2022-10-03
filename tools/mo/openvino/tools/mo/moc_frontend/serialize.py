# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

from openvino.runtime import Model  # pylint: disable=no-name-in-module,import-error

from openvino.tools.mo.back.ie_ir_ver_2.emitter import append_ir_info
from openvino.tools.mo.back.preprocessing import apply_preprocessing
from openvino.tools.mo.pipeline.common import get_ir_version
from openvino.tools.mo.utils.cli_parser import get_meta_info, parse_transform


def moc_emit_ir(ngraph_function: Model, argv: argparse.Namespace):
    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()

    # Apply preprocessing (mean/scale/reverse_channels/convert_layout/etc)
    apply_preprocessing(ov_function=ngraph_function, argv=argv)

    # Apply transformations
    from openvino.tools.mo.back.offline_transformations import apply_user_transformations, apply_moc_transformations, \
        apply_moc_legacy_transformations, apply_fused_names_cleanup

    apply_moc_transformations(ngraph_function)
    from openvino._offline_transformations import compress_quantize_weights_transformation
    compress_quantize_weights_transformation(ngraph_function)

    if argv.framework == "onnx":
        # set OldApi map in IR to be executed via OV API 1.x and for parity with legacy MO
        params_with_custom_types = [] if argv.placeholder_data_types is None \
            else list(argv.placeholder_data_types.keys())
        apply_moc_legacy_transformations(ngraph_function, params_with_custom_types)

    apply_user_transformations(ngraph_function, parse_transform(argv.transform))

    if argv.compress_fp16:
        from openvino.tools.mo.back.offline_transformations import compress_model
        compress_model(ngraph_function)

    apply_fused_names_cleanup(ngraph_function)

    del argv.feManager
    return ngraph_function
