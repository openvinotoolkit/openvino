import pytest
import tensorflow as tf
import random
from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shapes': [[random.randint(1, 4) for _ in range(2)], [random.randint(1, 4) for _ in range(2)]], 'axis': random.choice([-1,1])},
    {'shapes': [[random.randint(1, 30) for _ in range(3)], [random.randint(1, 30) for _ in range(3)], [random.randint(1, 30) for _ in range(3)]], 'axis': random.choice([-1,1])},
    {'shapes': [[random.randint(1, 30) for _ in range(4)], [random.randint(1, 30) for _ in range(4)], [random.randint(1, 30) for _ in range(4)], [random.randint(1, 30) for _ in range(4)]], 'axis': random.choice([-1,1])},
    {'shapes': [[random.randint(1, 30) for _ in range(2)], [random.randint(1, 30) for _ in range(2)]], 'axis': random.choice([-1,1])},
    {'shapes': [[random.randint(1, 4) for _ in range(3)], [random.randint(1, 30) for _ in range(3)], [random.randint(1, 30) for _ in range(3)]], 'axis': random.choice([-1,1])},
    {'shapes': [[random.randint(1, 30) for _ in range(4)], [random.randint(1, 30) for _ in range(4)], [random.randint(1, 30) for _ in range(4)], [random.randint(1, 30) for _ in range(4)]], 'axis': random.choice([-1,1])},
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
