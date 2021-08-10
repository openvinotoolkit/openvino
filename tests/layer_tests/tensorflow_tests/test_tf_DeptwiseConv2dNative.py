import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestDepthwiseConv2dNative(CommonTFLayerTest):

    def create_net(self, input_shape, filter_shape, ir_version):

        import tensorflow as tf
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            input_node = tf.compat.v1.placeholder(tf.float32, input_shape, 'Input')
            filter = np.reshape(np.array(range(np.prod(filter_shape))), filter_shape)
            filter = tf.convert_to_tensor(value=filter, dtype=tf.float32)
            conv_node = tf.nn.depthwise_conv2d(input=input_node, filter=filter, padding='VALID',
                                               strides=[1, 1, 1, 1])
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shape=[1, 64, 64, 1], filter_shape=[3, 3, 1, 2]),
        dict(input_shape=[1, 64, 64, 2], filter_shape=[3, 3, 2, 3]),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_add_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
