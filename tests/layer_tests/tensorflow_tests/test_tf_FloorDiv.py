# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestFloorDiv(CommonTFLayerTest):
    def create_add_placeholder_const_net(self, x_shape, dtype, ir_version, use_new_frontend):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(dtype, x_shape, 'Input')
            constant_value = np.array(-10).astype(dtype)
            y = tf.constant(constant_value)
            x = tf.raw_ops.Abs(x=x)
            res = tf.raw_ops.FloorDiv(x=x, y=y)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    # TODO: implement tests for 2 Consts + Add

    test_data_1D = [
        dict(x_shape=[], dtype=np.int32),
        dict(x_shape=[2], dtype=np.int64),
        dict(x_shape=[2, 4, 5], dtype=np.int32),
        dict(x_shape=[], dtype=np.float32),
        dict(x_shape=[2], dtype=np.float64),
        dict(x_shape=[2, 4, 5], dtype=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    @pytest.mark.precommit_tf_fe
    def test_add_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)


class TestFloorDivStaticInput(CommonTFLayerTest):
    stored_x_shape = []
    min = -100
    max = 200
    step = 1
    dtype = np.int32

    def create_flordiv_tf_net(self, min, max, step, y, dtype, ir_version, use_new_frontend, x_shape=None):
        import tensorflow as tf
        x = np.arange(min, max, step, dtype=dtype)
        
        # x_shape arg can contain -1, but we need always have positive values, therefore take concrete shape value after reshape
        x_shape = x.reshape(x_shape).shape if x_shape is not None else x.shape
        self.stored_x_shape = x_shape
        
        self.min = min
        self.max = max
        self.step = step
        self.dtype = dtype

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(dtype, x_shape, 'Input')
            y = tf.constant(np.array(y).astype(dtype))
            res = tf.raw_ops.FloorDiv(x=x, y=y)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net
    
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.arange(self.min, self.max, self.step, dtype=self.dtype)
            if self.stored_x_shape is not None:
                inputs_dict[input] = inputs_dict[input].reshape(self.stored_x_shape)
        return inputs_dict

    test_inputs = [
        # test for integers
        dict(min=-20, max=20, step=1, y=[10],dtype=np.int32),
        dict(min=-200, max=200, step=1, y=[-100], dtype=np.int32),
        dict(min=-1000, max=1000, step=2, y=[100], dtype=np.int32),
        dict(min=-10000, max=10000, step=100, y=[10000], dtype=np.int32),
        dict(min=-10000, max=10000, step=100, y=[-10000], dtype=np.int32),
        dict(min=-1e5, max=1e5, step=100, y=[1e5], dtype=np.int32),
        
        # test for multidimensinal input
        dict(min=-1000, max=1000, step=10, y=[1000], x_shape=[20, -1], dtype=np.int32),
        dict(min=-1000, max=1000, step=10, y=[1000], x_shape=[2, 5, -1], dtype=np.int32),
        dict(min=-1000, max=1000, step=1, y=[1000], x_shape=[2, 5, 10, -1], dtype=np.int32),

        # test for floats
        dict(min=-20, max=20, step=1, y=[10],dtype=np.float32),
        dict(min=-200, max=200, step=5, y=[200], dtype=np.float32),
        dict(min=-10000, max=10000, step=100, y=[10000], dtype=np.float32),
        dict(min=-1e5, max=1e5, step=100, y=[1e5], dtype=np.float32),
        dict(min=-1e5, max=1e5, step=100, y=[1e5], dtype=np.float64),
    ]
    @pytest.mark.parametrize("params", test_inputs)
    @pytest.mark.nightly
    @pytest.mark.precommit_tf_fe
    @pytest.mark.xfail(reason='CVS-132151: Layer tests should call ovc + save_model') 
    def test_floordiv(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend):
        self._test(*self.create_flordiv_tf_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                    use_new_frontend=use_new_frontend)
