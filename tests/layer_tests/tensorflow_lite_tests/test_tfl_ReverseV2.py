import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [1], 'axis': [-1]},
    {'shape': [1], 'axis': [0]},
    {'shape': [2, 6], 'axis': [-1, -2]},
    {'shape': [2, 6], 'axis': [1]},
    {'shape': [2, 4, 6], 'axis': [0, -2]},
    {'shape': [2, 4, 6], 'axis': [2]},
    {'shape': [2, 4, 6, 8], 'axis': [0, 3, -3, 2]},
    {'shape': [2, 4, 6, 8], 'axis': [-3]},
    {'shape': [2, 3, 1, 2, 2], 'axis': [0, 3, -3, 1, -1]},
    {'shape': [2, 3, 1, 2, 2], 'axis': [4]},
    {'shape': [2, 1, 1, 1, 2, 3, 2, 2], 'axis': [-1]},
    {'shape': [2, 1, 1, 1, 2, 3, 2, 2], 'axis': [0, 1, 2, 3, 4, 5, 6, 7]},
]


class TestTFLiteReverseV2LayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["ReverseV2"]
    allowed_ops = ['REVERSE_V2']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'axis'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            tf.reverse(place_holder, params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_reverse_v2(self, params, ie_device, precision, temp_dir):
        if len(params['axis']) > 1:
            pytest.xfail('CVS-109932')
        self._test(ie_device, precision, temp_dir, params)
