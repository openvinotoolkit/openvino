# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

from openvino.runtime import Model, PartialShape  # pylint: disable=no-name-in-module,import-error

from openvino.tools.mo.back.preprocessing import apply_preprocessing
from openvino.tools.mo.utils.cli_parser import parse_transform


def moc_emit_ir(ngraph_function: Model, argv: argparse.Namespace):
    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()
    if argv.framework == "pytorch":
        import torch
        from openvino.frontend.pytorch.decoder import pt_to_ov_type_map
        example_inputs = getattr(argv, "example_input", None)
        input_signature = getattr(argv, "input_signature", None)
        provide_shapes = argv.input_shape is not None
        if example_inputs is not None:
            inputs = [example_inputs] if isinstance(example_inputs, torch.Tensor) else example_inputs
            if input_signature is not None and isinstance(inputs, dict):
                ordered_inputs = []
                upd_sign = []
                for key in input_signature:
                    if key not in inputs:
                        continue
                    ordered_inputs.append(inputs[key])
                    upd_sign.append(key)
                inputs = ordered_inputs
                input_signature = upd_sign
            for idx, input_tensor in enumerate(ngraph_function.inputs):
                if isinstance(inputs, (list, tuple)):
                    input_data = inputs[idx]
                else:
                    input_data = list(inputs.values())[idx]
                pt_dtype = input_data.dtype if isinstance(input_data, torch.Tensor) else type(input_data)
                dtype = pt_to_ov_type_map.get(str(pt_dtype))
                if dtype is None:
                    raise f"Unknown input dtype {pt_dtype}"

                input_tensor.get_node().set_element_type(dtype)
                if input_signature is not None:
                    tensor = input_tensor.get_tensor()
                    input_names = tensor.names
                    input_names.update(input_signature[idx])
                    tensor.set_names(input_names)
                if not provide_shapes:
                    # prevent dynamic rank issue
                    shape = [-1] * len(input_data.shape)
                input_tensor.get_node().set_partial_shape(PartialShape(shape))
            ngraph_function.validate_nodes_and_infer_types() 

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
