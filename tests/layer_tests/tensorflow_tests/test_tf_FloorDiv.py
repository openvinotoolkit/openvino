# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import platform

from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()

def list_arm_platforms():
    return ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64']

class TestFloorDiv(CommonTFLayerTest):
    def create_add_placeholder_const_net(self, x_shape, dtype, ir_version, use_legacy_frontend):
        import tensorflow as tf
        self.dtype = dtype
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(dtype, x_shape, 'Input')
            constant_value = np.array(-10).astype(dtype)
            y = tf.constant(constant_value)
            res = tf.raw_ops.FloorDiv(x=x, y=y)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    def _prepare_input(self, inputs_info):
        tensor_name = list(inputs_info.keys())[0]
        x_shape = inputs_info[tensor_name]
        inputs_data = {}
        if np.issubdtype(self.dtype, np.floating):
            inputs_data[tensor_name] = rng.uniform(-5.0, 5.0, x_shape).astype(self.dtype)
        elif np.issubdtype(self.dtype, np.signedinteger):
            inputs_data[tensor_name] = rng.integers(-8, 8, x_shape).astype(self.dtype)
        else:
            inputs_data[tensor_name] = rng.integers(0, 8, x_shape).astype(self.dtype)
        return inputs_data

    # TODO: implement tests for 2 Consts + Add

    
    test_data_1D = [
        dict(x_shape=[], dtype=np.int32),
        dict(x_shape=[2], dtype=np.int64),
        dict(x_shape=[2, 4, 5], dtype=np.int32),
        dict(x_shape=[2, 4, 5], dtype=np.uint32),
        dict(x_shape=[2, 4, 5], dtype=np.uint64),

        dict(x_shape=[], dtype=np.float32),
        dict(x_shape=[2], dtype=np.float64),
        dict(x_shape=[2, 4, 5], dtype=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_add_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        if platform.system() == 'Linux' and platform.machine() in list_arm_platforms() and np.issubdtype(params['dtype'], np.signedinteger):
            pytest.xfail(reason='Ticket CVS-132377 - Divide inconsistent behavior on different systems')
        elif platform.system() == 'Darwin' and platform.machine() in list_arm_platforms():
            pytest.xfail(reason='Ticket - 132699')

        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestFloorDivStaticInput(CommonTFLayerTest):
    min = -100
    max = 200
    step = 1
    dtype = np.int32

    def create_flordiv_tf_net(self, min, max, step, y, dtype, ir_version, use_legacy_frontend):
        import tensorflow as tf
        x = np.arange(min, max, step, dtype=dtype)
        
        self.min = min
        self.max = max
        self.step = step
        self.dtype = dtype

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(dtype, x.shape, 'Input')
            y = tf.constant(np.array(y).astype(dtype))
            res = tf.raw_ops.FloorDiv(x=x, y=y)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net
    
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.arange(self.min, self.max, self.step, dtype=self.dtype)
        return inputs_dict

    test_inputs = [
        dict(min=-20, max=20, step=1, y=[10]),
        dict(min=-20, max=20, step=1, y=[5]),
        dict(min=-20, max=20, step=1, y=[6]),
        dict(min=-20, max=20, step=1, y=[-5]),
        dict(min=-20, max=20, step=1, y=[-6]),
        dict(min=-1e5, max=1e5, step=100, y=[1e5]),
    ]
    @pytest.mark.parametrize("params", test_inputs)
    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Linux' and platform.machine() in list_arm_platforms(),
                       reason='Ticket CVS-132377 - Divide inconsistent behavior on different systems')
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() in list_arm_platforms(),
                       reason='Ticket - 132699')
    def test_floordiv(self, params, dtype, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_flordiv_tf_net(**params, dtype=dtype, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                    use_legacy_frontend=use_legacy_frontend)
