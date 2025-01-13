# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging as log
import os
import torch

from e2e_tests.test_utils.pytorch_loaders import *
from e2e_tests.common.model_loader.provider import ClassProvider


class PyTorchModelLoader(ClassProvider):
    """PyTorch models loader runner."""
    __action_name__ = "load_pytorch_model"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self._config = config
        self.prepared_model = None

    def load_model(self, input_data):
        os.environ['TORCH_HOME'] = self._config.pop('torch_model_zoo_path')
        args = {k: v for k, v in self._config.items()}
        module = args['import-module']
        try:
            log.info('Preparing model for MO ...')
            pytorch_loader = LoadPyTorchModel(module=module,
                                              args=args,
                                              inputs=input_data)
            self.prepared_model = pytorch_loader.load_model()
            if args['weights']:
                self.prepared_model.load_state_dict(torch.load(args['weights'], map_location='cpu'))
        except Exception as err:
            raise Exception from err

        return self.prepared_model


class CustomPytorchModelLoader(ClassProvider):
    __action_name__ = "custom_pytorch_model_loader"

    def __init__(self, config):
        self.execution_function = config["execution_function"]
        self.prepared_model = None

    def load_model(self, data):
        self.prepared_model = self.execution_function(data)
        return self.prepared_model
