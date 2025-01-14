# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tempfile
import tensorflow as tf
from common import constants
from pathlib import Path


def create_net_list(input_names, input_shapes):
    tf.keras.backend.clear_session()

    # Create TensorFlow 2 model with multiple outputs.
    # Outputs are list.

    inputs = []
    outputs = []
    for ind in range(len(input_names)):
        input = tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind])
        inputs.append(input)
        outputs.append(tf.keras.layers.Activation(tf.nn.sigmoid)(input))

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_net_dict(input_names, input_shapes):
    tf.keras.backend.clear_session()

    # Create TensorFlow 2 model with multiple outputs.
    # Outputs are dictionary.

    inputs = []
    outputs = {}
    for ind in range(len(input_names)):
        input = tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind])
        inputs.append(input)
        outputs["name" + str(ind)] = tf.keras.layers.Activation(tf.nn.sigmoid)(input)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def check_outputs_by_order(fw_output, ov_output, eps):
    # Compare outputs by indices
    for idx, output in enumerate(fw_output):
        fw_out = output.numpy()
        ov_out = ov_output[idx]
        assert fw_out.shape == ov_out.shape, "Output with index {} has shape different from original FW.".format(idx)
        diff = np.max(np.abs(fw_out - ov_out))
        assert diff < eps, "Output with index {} has inference result different from original FW.".format(idx)


def check_outputs_by_names(fw_output, ov_output, eps):
    # Compare outputs by names
    for name, output in fw_output.items():
        fw_out = output.numpy()
        ov_out = ov_output[name]
        assert fw_out.shape == ov_out.shape, "Output with name {} has shape different from original FW.".format(name)
        diff = np.max(np.abs(fw_out - ov_out))
        assert diff < eps, "Output with name {} has inference result different from original FW.".format(name)


class TestTFInputOutputOrder():
    def setup_method(self):
        Path(constants.out_path).mkdir(parents=True, exist_ok=True)
        self.tmp_dir = tempfile.TemporaryDirectory(dir=constants.out_path).name

    @pytest.mark.parametrize("save_to_file, create_model_method, compare_model_method", [
        (False, 'create_net_list', 'check_outputs_by_order'),
        (False, 'create_net_dict', 'check_outputs_by_names'),
        # next two cases are failing due to TensorFlow bug https://github.com/tensorflow/tensorflow/issues/75177
        pytest.param(True, 'create_net_list', 'check_outputs_by_order',
                     marks=pytest.mark.xfail(reason='https://github.com/tensorflow/tensorflow/issues/75177')),
        pytest.param(True, 'create_net_dict', 'check_outputs_by_names',
                     marks=pytest.mark.xfail(reason='https://github.com/tensorflow/tensorflow/issues/75177')),
    ])
    def test_order(self, ie_device, precision, save_to_file, create_model_method, compare_model_method):
        from openvino import convert_model, compile_model
        input_names = ["k", "b", "m", "c", "x"]
        input_shapes = [[1, 1], [1, 3], [1, 2], [1, 5], [1, 4]]
        epsilon = 0.001

        fw_model = eval(create_model_method)(input_names, input_shapes)

        if save_to_file:
            fw_model.export(self.tmp_dir + "./model")
            ov_model = convert_model(self.tmp_dir + "./model")
        else:
            ov_model = convert_model(fw_model)

        cmp_model = compile_model(ov_model, ie_device)
        test_inputs = []
        for shape in input_shapes:
            test_inputs.append(np.random.rand(*shape))

        fw_output = fw_model(test_inputs)
        ov_output = cmp_model(test_inputs)

        eval(compare_model_method)(fw_output, ov_output, epsilon)
