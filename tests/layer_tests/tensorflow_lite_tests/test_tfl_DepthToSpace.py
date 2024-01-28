import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [random.randint(1, 20) for _ in range(4)], 'block_size': random.randint(1,2)},
    {'shape': [random.randint(1, 50) for _ in range(4)], 'block_size': random.randint(1,5)},
]


class TestTFLiteDepthToSpaceLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["DepthToSpace"]
    allowed_ops = ['DEPTH_TO_SPACE']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'block_size'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            tf.nn.depth_to_space(place_holder, params['block_size'], 'NHWC', name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_depth_to_space(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
