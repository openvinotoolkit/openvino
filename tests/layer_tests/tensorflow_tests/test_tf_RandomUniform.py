import pytest
import tensorflow as tf

from common.tf_layer_test_class import CommonTFLayerTest


class TestTFRandomUniform(CommonTFLayerTest):
    def create_tf_random_uniform_net(self, global_seed, op_seed, x_shape, min_val, max_val, input_type, ir_version):
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = x_shape.copy()
            # reshaping
            if len(tf_x_shape) >= 3:
                tf_x_shape.append(tf_x_shape.pop(1))

            x = tf.compat.v1.placeholder(input_type, x_shape, 'Input')
            if global_seed is not None:
                tf.random.set_seed(global_seed)
            random_uniform = tf.random.uniform(x_shape, seed=op_seed, dtype=input_type, minval=min_val,
                                               maxval=max_val) + x

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf_net, ref_net

    test_data = [pytest.param(
        dict(global_seed=32465, op_seed=48971, min_val=0.0, max_val=1.0, x_shape=[3, 7], input_type=tf.float32),
        marks=pytest.mark.precommit),
                 dict(global_seed=None, op_seed=56197, min_val=-100, max_val=100, x_shape=[6], input_type=tf.float32),
                 dict(global_seed=78132, op_seed=None, min_val=-200, max_val=-50, x_shape=[5, 8], input_type=tf.int32),
                 dict(global_seed=4571, op_seed=48971, min_val=1.5, max_val=2.3, x_shape=[7], input_type=tf.float32),
                 dict(global_seed=32465, op_seed=12335, min_val=-150, max_val=-100, x_shape=[18], input_type=tf.int32)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.xfail(reason="Needs RandomUniform reference implementation. Ticket: 56596.")
    def test_tf_random_uniform(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip("RandomUniform is not supported on GPU")
        self._test(*self.create_tf_random_uniform_net(**params, ir_version=ir_version), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, **params)
