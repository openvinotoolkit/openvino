import itertools

import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import data_generators

test_ops = [
    {'op_name': ['RESIZE_BILINEAR'], 'op_func': tf.compat.v1.image.resize_bilinear},
]

test_params = [
    {'shape': [1, 3, 4, 3], 'size': [1, 1], 'align_corners': True, 'half_pixel_centers': False},
    {'shape': [1, 3, 4, 3], 'size': [1, 1], 'align_corners': False, 'half_pixel_centers': False},
    {'shape': [1, 3, 4, 3], 'size': [1, 1], 'align_corners': False, 'half_pixel_centers': True},
    {'shape': [1, 16, 24, 3], 'size': [8, 12], 'align_corners': True, 'half_pixel_centers': False},
    {'shape': [1, 16, 24, 3], 'size': [8, 12], 'align_corners': False, 'half_pixel_centers': False},
    {'shape': [1, 16, 24, 3], 'size': [8, 12], 'align_corners': False, 'half_pixel_centers': True},
]


test_data = list(itertools.product(test_ops, test_params))
for i, (parameters, shapes) in enumerate(test_data):
    parameters.update(shapes)
    test_data[i] = parameters.copy()


class TestTFLiteReshapeLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["ResizeBilinear"]

    def _prepare_input(self, inputs_dict, generator=None):
        if generator is None:
            return super()._prepare_input(inputs_dict)
        return data_generators[generator](inputs_dict)

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'size', 'align_corners',
                                                    'half_pixel_centers'})) == 6, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get("dtype", tf.float32), params["shape"],
                                                   name=self.inputs[0])
            params['op_func'](placeholder, size=params['size'],
                              align_corners=params['align_corners'],
                              half_pixel_centers=params['half_pixel_centers'],
                              name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_resize_bilinear(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
