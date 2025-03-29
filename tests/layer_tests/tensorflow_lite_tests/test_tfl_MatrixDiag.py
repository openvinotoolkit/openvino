import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [3]},
    {'shape': [1, 22]},
    {'shape': [1, 1, 8]},
    {'shape': [1, 22, 22, 8]},
    {'shape': [1, 22, 22, 8, 3]},
]


class TestTFLiteMatrixDiagLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["MatrixDiag"]
    allowed_ops = ['MATRIX_DIAG']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape'})) == 1, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            place_holder = tf.compat.v1.placeholder(params.get('dtype', tf.float32), params['shape'],
                                                    name=self.inputs[0])
            tf.linalg.diag(place_holder, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_matrix_diag(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
