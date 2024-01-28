import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import additional_test_params
from common.utils.tflite_utils import parametrize_tests

test_params = [
    {'shape': [random.randint(1, 10) for _ in range(1)]},
    {'shape': [random.randint(1, 10) for _ in range(2)]},
    {'shape': [random.randint(1, 10) for _ in range(3)]},
    {'shape': [random.randint(1, 10) for _ in range(4)]}
]

test_data = parametrize_tests(test_params, additional_test_params[0])


class TestTFLiteExpandDimsLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["L2_Normalization"]
    allowed_ops = ['L2_NORMALIZATION']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape'})) == 1, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            tf.math.l2_normalize(place_holder, axis=params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_l2_normalization(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
