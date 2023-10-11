import pytest
import tensorflow as tf
import numpy as np

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shapes': ((None, 4, 5), (None, 5, 6), (3, 4, 5), (3, 5, 6)), 'adjoint_a': True, 'adjoint_b': True},
    {'shapes': ((None, 1, 3, 4), (None, 4, 2), (2, 1, 3, 4), (5, 4, 2)), 'adjoint_a': False, 'adjoint_b': False},
    {'shapes': ((None, None, None, 3, 4), (None, None, None, 4, 3),
                (2, 2, 2, 3, 4), (2, 2, 2, 4, 3)), 'adjoint_a': False, 'adjoint_b': False},
]


class TestTFLiteBatchMatmulLayerTest(TFLiteLayerTest):
    inputs = ["Input", "Input1"]
    outputs = ["BatchMatmul"]
    allowed_ops = ['BATCH_MATMUL']

    def _prepare_input(self, inputs_dict, generator=None):
        input0_shape = self.shapes[2]
        adj_a = self.adjoint_a
        adj_b = self.adjoint_b
        if adj_a:
            input0_shape = self._swap_last_two_dims(*input0_shape)
        inputs_dict['Input'] = np.float32((1.0 - (-1.0)) * np.random.random_sample(input0_shape) + (-1.0))

        input1_shape = self.shapes[3] if not adj_b else self._swap_last_two_dims(*self.shapes[3])
        inputs_dict['Input1'] = np.float32((1.0 - (-1.0)) * np.random.random_sample(input1_shape) + (-1.0))

        return inputs_dict

    def _swap_last_two_dims(self, *args):
        """Return a tuple with the last two dimensions swapped."""
        return args[:-2] + (args[-1],) + (args[-2],)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shapes', 'adjoint_a', 'adjoint_b'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.shapes = params['shapes']
        self.adjoint_a = params['adjoint_a']
        self.adjoint_b = params['adjoint_b']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder0_shape = self.shapes[0]
            adj_a = params["adjoint_a"]
            adj_b = params["adjoint_b"]

            if adj_a:
                placeholder0_shape = self._swap_last_two_dims(*placeholder0_shape)
            input0_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=placeholder0_shape, name=self.inputs[0])
            if adj_b:
                placeholder1_shape = self._swap_last_two_dims(*self.shapes[1])
            else:
                placeholder1_shape = self.shapes[1]
            input1_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=placeholder1_shape, name=self.inputs[1])

            tf.matmul(input0_tensor, input1_tensor, adjoint_a=adj_a, adjoint_b=adj_b, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_batch_matmul(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
