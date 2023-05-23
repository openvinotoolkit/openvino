import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [5, 1], 'indices_shape': [1, 1], 'batch_dims': 0},

    {'shape': [5, 5], 'indices_shape': [2, 1], 'batch_dims': 0},
    {'shape': [5, 5], 'indices_shape': [2, 2], 'batch_dims': 0},
    #
    {'shape': [3, 5, 10], 'indices_shape': [3, 2], 'batch_dims': 1},
    {'shape': [5, 5, 10], 'indices_shape': [2, 3], 'batch_dims': 0},
    {'shape': [4, 5, 10], 'indices_shape': [2, 10, 1], 'batch_dims': 2},
    ]


class TestTFLiteGatherLayerTest(TFLiteLayerTest):
    inputs = ["Input_x"]
    outputs = ["GatherND"]
    allowed_ops = ['GATHER_ND']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'indices_shape', 'batch_dims'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        self.outputs = ["GatherND"]
        self.allowed_ops = ["GATHER_ND"]
        if params['batch_dims'] > 0:
            output_name = "GatherND"
            self.outputs = [f"GatherND/Reshape_{params['batch_dims'] + 2}"]
            if params['batch_dims'] > 1:
                self.allowed_ops.append('RESHAPE')
        else:
            output_name = self.outputs[0]

        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get('dtype', tf.float32),
                                                   params['shape'], name=self.inputs[0])
            constant = tf.constant(np.random.randint(0, params['shape'][0] - 1, params['indices_shape']))

            tf.gather_nd(placeholder, constant, name=output_name, batch_dims=params['batch_dims'])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_gather_nd(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
