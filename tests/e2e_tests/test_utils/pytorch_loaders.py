# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import sys

import torch


class LoadPyTorchModel:
    def __init__(self, module: str, args: dict, inputs: dict):
        self.module = module
        self.args = args
        self.export = args.get('torch_export_method')
        self.inputs_order = args.get('inputs_order')
        self.inputs = inputs
        self.model = None

    def load_model(self):
        self.model = loader_map[self.module](self.module, self.args)
        self.model.eval()
        if self.export == 'trace':
            self.inputs = self._convert_inputs()
            self.model = self._trace()
        if self.export == 'export':
            self.model = self._script()
        return self.model

    def _trace(self):
        assert self.model, "Model should be loaded through 'load_model"
        self.model = torch.jit.trace(self.model, self.inputs, strict=False)
        return self.model

    def _script(self):
        assert self.model, "Model should be loaded through 'load_model"
        self.model = torch.jit.script(self.model, self.inputs)
        return self.model

    def _convert_inputs(self):
        helper = []
        if self.inputs_order:
            for input_name in self.inputs_order:
                helper.append(self.inputs[input_name])
        else:
            helper = list(self.inputs.values())

        return helper


def load_torchvision_model(module, args):
    module = importlib.import_module(module)

    creator = getattr(module, args['model-name'])
    model = creator(**args['model-param'], pretrained=True)

    return model


def load_cadene_model(module, args):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    module = importlib.import_module(module)
    creator = getattr(module, args['model-name'])
    model = creator(**args['model-param'])

    return model


def load_hugging_face_model(module, args):
    module = importlib.import_module(module)
    model = module.AutoModel.from_pretrained(args['model-name'], torchscript=True)

    return model


def load_timm_model(module, args):
    module = importlib.import_module(module)
    model = module.create_model(args['model-name'], pretrained=True)

    return model


def load_saved_jit_model(module, args):
    module = importlib.import_module('torch')
    return module.jit.load(args['model-path'])


def load_saved_model(module, args):
    if args.get('model_class_path'):
        sys.path.insert(0, os.path.abspath(args['model_class_path']))
    module = importlib.import_module(module)
    return module.load(args['model-path'])


loader_map = {
    'torchvision.models': load_torchvision_model,
    'torchvision.models.detection': load_torchvision_model,
    'torchvision.models.optical_flow': load_torchvision_model,
    'pretrainedmodels': load_cadene_model,
    'pretrained': load_cadene_model,
    'transformers': load_hugging_face_model,
    'timm': load_timm_model,
    'torch_jit': load_saved_jit_model,
    'torch': load_saved_model
}
