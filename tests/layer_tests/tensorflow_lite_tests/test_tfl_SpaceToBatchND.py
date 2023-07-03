import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [8, 10, 10, 3], 'block_shape': [2, 2], 'paddings': [[0, 2], [0, 0]]},
    {'shape': [24, 10, 10, 1], 'block_shape': [6, 6], 'paddings': [[2, 0], [0, 2]]}
]


class TestTFLiteSpaceToBatchNDLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["SpaceToBatchOP"]
    allowed_ops = ['SPACE_TO_BATCH_ND']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'block_shape', 'paddings'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            tf.space_to_batch(place_holder, params['block_shape'], params['paddings'],
                              name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_space_to_batch_nd(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
