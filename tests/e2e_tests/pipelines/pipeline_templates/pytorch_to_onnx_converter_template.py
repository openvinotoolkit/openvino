# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def convert_pytorch_to_onnx(model_name=None,
                            weights=None,
                            input_shapes=None,
                            output_file=None,
                            model_path=None,
                            import_module=None,
                            input_names=None,
                            output_names=None,
                            model_param=None,
                            inputs_dtype=None,
                            conversion_param=None,
                            opset_version=None,
                            torch_model_zoo_path='',
                            converter_timeout=300):
    return "pytorch_to_onnx", {"convert_pytorch_to_onnx": {
        "model-name": model_name,
        "weights": weights,
        "input-shapes": input_shapes,
        "output-file": output_file,
        "model-path": model_path,
        "import-module": import_module,
        "input-names": input_names,
        "output-names": output_names,
        "model-param": model_param,
        "inputs-dtype": inputs_dtype,
        "conversion-param": conversion_param,
        "opset_version": opset_version,
        "torch_model_zoo_path": torch_model_zoo_path,
        "converter_timeout": converter_timeout,
    }}
