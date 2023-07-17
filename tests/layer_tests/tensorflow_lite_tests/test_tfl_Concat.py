import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shapes': [[1, 3], [1, 7]], 'axis': -1},
    {'shapes': [[1, 3, 10], [1, 3, 56], [1, 3, 2]], 'axis': -1},
    {'shapes': [[1, 3, 10, 17], [1, 3, 10, 1], [1, 3, 10, 5], [1, 3, 10, 3]], 'axis': -1},
    {'shapes': [[2, 7], [2, 8]], 'axis': 1},
    {'shapes': [[1, 3, 10], [1, 2, 10], [1, 5, 10]], 'axis': 1},
    {'shapes': [[1, 3, 10, 17], [1, 10, 10, 17], [1, 12, 10, 17], [1, 1, 10, 17]], 'axis': 1},
]


class TestTFLiteConcatTest(TFLiteLayerTest):
    outputs = ['Concat']
    allowed_ops = ['CONCATENATION']

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shapes', 'axis'})) == 2, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholders = []
            self.inputs = []
            for j, placeholder_shape in enumerate(params['shapes']):
                concat_input = f'concat_input_{j}'
                self.inputs.append(concat_input)
                placeholders.append(tf.compat.v1.placeholder(params.get('dtype', tf.float32), placeholder_shape,
                                                             name=concat_input))
            tf.concat(placeholders, params['axis'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_concat(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
