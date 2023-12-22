import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [2], 'k': 2, 'sorted': True},
    {'shape': [2, 3], 'k': 1, 'sorted': False},
    {'shape': [2, 3, 5], 'k': 2, 'sorted': True},
    {'shape': [2, 3, 5, 10], 'k': 9, 'sorted': False},
]


class TestTFLiteTopKV2LayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["TopKV2"]
    allowed_ops = ['TOPK_V2']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'k', 'sorted'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(tf.float32, params['shape'], self.inputs[0])
            tf.raw_ops.TopKV2(input=placeholder, k=params['k'], sorted=params['sorted'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_topk_v2(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
