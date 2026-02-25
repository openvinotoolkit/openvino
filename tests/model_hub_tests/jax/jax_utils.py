# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import flax
import jax
import numpy as np
from models_hub_common.test_convert_model import TestConvertModel
from openvino import convert_model


def flattenize_pytree(outputs):
    leaves, _ = jax.tree_util.tree_flatten(outputs)
    return [np.array(i) if isinstance(i, jax.Array) else i for i in leaves]


class TestJaxConvertModel(TestConvertModel):
    def get_inputs_info(self, _):
        return None

    def prepare_inputs(self, _):
        inputs = getattr(self, 'inputs', self.example)
        return inputs

    def convert_model(self, model_obj):
        if isinstance(model_obj, flax.linen.Module):
            ov_model = convert_model(model_obj, example_input=self.example, verbose=True)
        else:
            # create JAXpr object
            jaxpr = jax.make_jaxpr(model_obj)(**self.example)
            ov_model = convert_model(jaxpr, verbose=True)
        return ov_model

    def infer_fw_model(self, model_obj, inputs):
        if isinstance(inputs, dict):
            fw_outputs = model_obj(**inputs)
        elif isinstance(inputs, list):
            fw_outputs = model_obj(*inputs)
        else:
            fw_outputs = model_obj(inputs)
        return flattenize_pytree(fw_outputs)
