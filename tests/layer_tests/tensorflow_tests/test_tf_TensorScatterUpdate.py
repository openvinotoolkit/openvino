import numpy as np
import tensorflow as tf
import pytest
from common.tf_layer_test_class import CommonTFLayerTest

class TestTFTensorScatterUpdate(CommonTFLayerTest):
    def create_tf_tensor_scatter_update_net(self, x_shape, indices, updates):
        tf.compat.v1.reset_default_graph()

        # Create placeholder and constant tensors
        x = tf.compat.v1.placeholder(tf.float32, x_shape, 'Input')
        tf_indices = tf.constant(indices)
        tf_updates = tf.constant(updates)

        # Apply TensorScatterUpdate operation
        updated_tensor = tf.tensor_scatter_update(x, tf_indices, tf_updates)

        # Initialize variables and get the graph definition
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            tf_net = sess.graph_def

        return tf_net

    test_data = [
        pytest.param(
            dict(x_shape=[8], indices=[4, 3, 1, 7], updates=[9.0, 10.0, 11.0, 12.0]),
            marks=pytest.mark.precommit),
        pytest.param(dict(x_shape=[4, 4, 4], indices=[[0], [2]], updates= \
            [[[5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0], [7.0, 7.0, 7.0, 7.0],
              [8.0, 8.0, 8.0, 8.0]],
             [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0],
              [4.0, 4.0, 4.0, 4.0]]])),
        pytest.param(dict(x_shape=[2, 2], indices=[[0]], updates=[[5.0, 3.0]])),
        pytest.param(dict(x_shape=[2, 2], indices=[[1, 1]], updates=[5.0])),
        dict(x_shape=[1], indices=[[0]], updates=[3.0]),
        dict(x_shape=[20], indices=[[0], [6], [9], [19], [13]],
             updates=[3.0, 7.0, -12.0, 4.0, -99.0]),
        dict(x_shape=[4, 2], indices=[[1], [2]], updates=[[9.0, 14.0], [-76.0, 0.0]]),
        dict(x_shape=[4, 4, 4], indices=[[0], [1], [3]], updates=[
            [[5.0, 1.0, 5.0, 13.0], [8.0, 6.0, 6.0, 8.0], [7.0, 0.0, 0.0, 7.0],
             [8.0, 8.0, 8.0, 8.0]],
            [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0]],
            [[5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0], [7.0, 7.0, 7.0, 7.0],
             [8.0, 8.0, 8.0, 8.0]]]),
        dict(x_shape=[2, 2, 2], indices=[[1, 1, 1], [0, 1, 0]], updates=[9.0, 6.3]),
        pytest.param(dict(x_shape=[2, 2, 2], indices=[[0, 0], [0, 1]], updates=[[6.7, 9.0], [45.0, 8.3]]),
                     marks=pytest.mark.precommit_tf_fe),
        dict(x_shape=[2, 2, 2], indices=[[1]], updates=[[[6.7, 9.0], [45.0, 8.3]]]),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_tf_tensor_scatter_update(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_tf_tensor_scatter_update_net(**params),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)