import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [8, 10, 10, 16], 'block_size': 2},
    {'shape': [24, 10, 10, 50], 'block_size': 5},
]


class TestTFLiteSpaceToDepthLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["SpaceToDepth"]
    allowed_ops = ['SPACE_TO_DEPTH']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'block_size'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            tf.nn.space_to_depth(place_holder, params['block_size'], 'NHWC', name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_space_to_depth(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
