# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.tools import flatbuffer_utils as utils

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [2, 10]},
    {'shape': [2, 10, 10, 3]},
]


class TestTFLiteRelu0To1LayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Relu0To1"]
    allowed_ops = ['RELU_0_TO_1']

    def _prepare_input(self, inputs_dict, generator=None):
        inputs_dict['Input'] = np.float32(4 * np.random.random_sample(inputs_dict['Input']) - 2)
        return inputs_dict

    def make_model(self, params):
        assert 'shape' in params, 'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(tf.float32, params['shape'],
                                                   name=self.inputs[0])
            # Use RELU as a placeholder; produce_tflite_model will replace it with RELU_0_TO_1
            tf.nn.relu(placeholder, name=self.outputs[0])
            net = sess.graph_def
        return net

    def produce_tflite_model(self, framework_model, save_path):
        # Convert the TF graph (with RELU) to TFLite
        tflite_model_path = super().produce_tflite_model(framework_model, save_path)

        # Post-process: replace RELU with RELU_0_TO_1 in the flatbuffer.
        model_obj = utils.read_model(tflite_model_path)
        replaced = False
        relu0to1_op = schema_fb.BuiltinOperator.RELU_0_TO_1
        placeholder = schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES
        for oc in model_obj.operatorCodes:
            if (oc.builtinCode == schema_fb.BuiltinOperator.RELU or
                    oc.deprecatedBuiltinCode == schema_fb.BuiltinOperator.RELU):
                oc.builtinCode = relu0to1_op
                oc.deprecatedBuiltinCode = min(relu0to1_op, placeholder)
                replaced = True

        assert replaced, "Failed to find RELU operator to replace with RELU_0_TO_1"

        relu0to1_path = os.path.join(save_path, 'relu_0_to_1.tflite')
        utils.write_model(model_obj, relu0to1_path)
        return relu0to1_path

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_relu_0_to_1(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
