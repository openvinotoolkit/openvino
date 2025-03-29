import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [1, 3], 'axis': [0]},
    {'shape': [2, 1], 'axis': [1]},
    {'shape': [1, 1, 2], 'axis': [0, 1]},
    {'shape': [1, 1, 2, 2], 'axis': [1]},
    {'shape': [1, 1, 2, 2], 'axis': [1, 1]},
    {'shape': [5, 1, 1, 1], 'axis': [-1, -2, -3]},
]


class TestTFLiteSqueezeLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Squeeze"]
    # TFLite returns SQUEEZE only when it has undetermined rank, but OV doesn't support SQUEEZE op with such rank
    allowed_ops = ['RESHAPE']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'axis'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            input_value1 = tf.compat.v1.placeholder(tf.float32, params['shape'], name=self.inputs[0])
            tf.squeeze(input_value1, params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_squeeze_dims(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
