# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from openvino.tools.pot import Metric, DataLoader, IEEngine, \
    load_model, save_model, compress_model_weights, create_pipeline

class QuantizationParameters(object):
    """
    Basic parameters of the quantization methods    
    """
    def __init__(self):
        """
        Initializes quantization parameters with some reasonable verified defaults.
        """
        self._model_name = "model"
        self._target_device = "ANY"
        self._method_name = "DefaultQuantization"
        self._preset = "performance"
        self._subset_size = 300
        self._model_type = None

    @property
    def target_device(self) -> str:
        """
        :return: The target device to quantize for.   
        """
        return self._target_device

    @property
    def method_name(self) -> str:
        """
        :return: The name of the quantization method.
        """
        return self._method_name

    @property
    def model_name(self) -> str:
        """
        :return: (Optional) The name of the model.   
        """
        return self._model_name

    @property
    def model_path(self) -> str:
        """
        :return: Path to the model description (.xml).
        """
        return self._model_path

    @property
    def weights_path(self) -> str:
        """
        :return: Path to the model weights (.bin).
        """
        return self._weights_path

    @property
    def preset(self) -> str:
        """
        :return: (Optional) Quantization preset: p
            "performance" (default) - for symmetric quantization
            "mixed" - for mixed scheme
        """
        return self._preset

    @property
    def subset_size(self) -> int:
        """
        :return: (Optional) Size of the dataset for statistics collection (300 by default).
        """
        return self._subset_size

    @property
    def model_type(self) -> str:
        """
        :return: (Optional) Model type. The only possible value is "transformers"
                 for quantization of Transformers-family model.
        """
        return self._model_type

def quantize_post_training(parameters: QuantizationParameters, data_loader: DataLoader):
    """
    Simple API for model quantization that  requires minimum set of parameters to be specified.
    Basic steps include:
    # Step 1: Create the data loader object.
    data_loader = UserDataLoader(params) # user-defined DataLoader

    # Step 2: Use a helper for post-training quantization - quantize_post_training
    optimized_model = quantize_post_training(params, data_loader)

    # Step 3: Export optimized model.
    export(optimized_model, path, options) # Serialization to IR / export to ONNX or TF Frozen graph
    """

    model_config = {
        'model_name': parameters.model_name,
        'model': os.path.expanduser(parameters.model_path),
        'weights': os.path.expanduser(parameters.weights_path)
    }

    engine_config = {
        'device': 'CPU'
    }

    algorithms = [
        {
            'name': parameters.method_name,
            'params': {
                'target_device': parameters.target_device,
                'preset': parameters.preset,
                'stat_subset_size': parameters.subset_size
            }
        }
    ]

    # Step 1: Load the model.
    model = load_model(model_config)

    # Step 2: Initialize the engine
    engine = IEEngine(config=engine_config,
                      data_loader=data_loader)

    # Step 3: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(algorithms, engine)

    # Step 4: Execute the pipeline.
    compressed_model = pipeline.run(model)

    return compressed_model

class ExportParameters(object):
    """
    Optimized model exports specific parameters
    """
    def __init__(self):
        self._compress_weights = True

    @property
    def compress_weights(self) -> bool:
        """
        :return: (Optional) Whether to compress weights to the target precision.
        """
        return self._compress_weights


def export(compressed_model, path_to_save, parameters: ExportParameters = ExportParameters()) -> None:
    """
    Basic parameters of the quantization methods    
    """
    if parameters.compress_weights:
        compress_model_weights(compressed_model)

    save_model(compressed_model, os.path.expanduser(path_to_save))