# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def pytorch_loader(model_name=None,
                   weights=None,
                   output_file=None,
                   model_path=None,
                   import_module=None,
                   model_param=None,
                   inputs_order=None,
                   torch_export_method='',
                   torch_model_zoo_path='',
                   model_class_path='',
                   loader_timeout=300):
    return "load_model", {"load_pytorch_model": {
        "model-name": model_name,
        "weights": weights,
        "output-file": output_file,
        "model-path": model_path,
        "import-module": import_module,
        "model-param": model_param,
        "torch_export_method": torch_export_method,
        "inputs_order": inputs_order,
        "torch_model_zoo_path": torch_model_zoo_path,
        "model_class_path": model_class_path,
        "loader_timeout": loader_timeout,
    }}


def custom_pytorch_loader(func: callable, *args, **kwargs):
    return 'load_model', {
        'custom_pytorch_model_loader': {"execution_function": func(*args, **kwargs)}}


def tf_hub_loader(model_name=None,
                  model_link=None):
    return "tf_hub_load_model", {"load_tf_hub_model": {
        "model_name": model_name,
        'model_link': model_link,
    }}
