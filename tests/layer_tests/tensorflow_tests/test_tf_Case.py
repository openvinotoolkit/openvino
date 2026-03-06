import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from tensorflow.python.framework import function

rng = np.random.default_rng(32545)

class TestRawCaseOp(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'cond:0' in inputs_info
        assert 'input_data:0' in inputs_info
        cond_shape = inputs_info['cond:0']
        input_shape = inputs_info['input_data:0']
        inputs_data = {}
        inputs_data['cond:0'] = rng.integers(0, 3, cond_shape).astype(np.int32)
        inputs_data['input_data:0'] = rng.integers(1, 10, input_shape).astype(np.int32)
        return inputs_data

    def create_case_net(self, input_shape, cond_value):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            cond = tf.compat.v1.placeholder(dtype=tf.int32, shape=(), name="cond")
            input_data = tf.compat.v1.placeholder(dtype=tf.int32, shape=input_shape, name="input_data")

            @function.Defun()
            def branch_fn_1():
                return tf.add(input_data, tf.constant(1, dtype=tf.int32))
            @function.Defun()
            def branch_fn_2():
                return tf.multiply(input_data, tf.constant(2, dtype=tf.int32))
            @function.Defun()
            def branch_fn_3():
                return tf.multiply(input_data, tf.constant(3, dtype=tf.int32))
            @function.Defun()
            def branch_fn_4():
                return tf.multiply(input_data, tf.constant(4, dtype=tf.int32))
            branches_fn = [branch_fn_1, branch_fn_2, branch_fn_3, branch_fn_4]

            tf.raw_ops.Case(branch_index=cond, input=[input_data], branches=branches_fn, Tout=[tf.int32])
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[1, 2], cond_value=0),
        dict(input_shape=[4, 2], cond_value=1),
        dict(input_shape=[6, 6], cond_value=2),
        dict(input_shape=[1, 5], cond_value=3),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_Case_op(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_case_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)


