import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': ['MATRIX_DIAG'], 'op_func': tf.linalg.diag},
]

test_params = [
    {'shape': [3]},
    {'shape': [1, 22]},
    {'shape': [1, 1, 8]},
    {'shape': [1, 22, 22, 8]},
    {'shape': [1, 22, 22, 8, 3]},
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteMatrixDiagLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["MatrixDiag"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape'})) == 3, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            params['op_func'](place_holder, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_matrix_diag(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
