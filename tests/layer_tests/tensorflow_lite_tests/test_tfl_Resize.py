import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest
from common.utils.tflite_utils import parametrize_tests

test_ops = [
    {'op_name': ['RESIZE_BILINEAR'], 'op_func': 'tf.compat.v1.image.resize_bilinear'},
    {'op_name': ['RESIZE_NEAREST_NEIGHBOR'], 'op_func': 'tf.compat.v1.image.resize_nearest_neighbor'},
]

test_params = [
    {'shape': [1, 3, 4, 3], 'size': [1, 1], 'align_corners': True, 'half_pixel_centers': False},
    {'shape': [1, 3, 4, 3], 'size': [1, 1], 'align_corners': False, 'half_pixel_centers': False},
    {'shape': [1, 3, 4, 3], 'size': [1, 1], 'align_corners': False, 'half_pixel_centers': True},  # accuracy failure

    {'shape': [1, 3, 4, 3], 'size': [10, 10], 'align_corners': True, 'half_pixel_centers': False},
    {'shape': [1, 3, 4, 3], 'size': [10, 10], 'align_corners': False, 'half_pixel_centers': False},
    {'shape': [1, 3, 4, 3], 'size': [10, 10], 'align_corners': False, 'half_pixel_centers': True},  # accuracy failure
]

test_data = parametrize_tests(test_ops, test_params)


class TestTFLiteResizeLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["Resize"]

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'op_name', 'op_func', 'shape', 'size', 'align_corners',
                                                    'half_pixel_centers'})) == 6, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.allowed_ops = params['op_name']
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params.get("dtype", tf.float32), params["shape"],
                                                   name=self.inputs[0])
            eval(params['op_func'])(placeholder, size=params['size'],
                              align_corners=params['align_corners'],
                              half_pixel_centers=params['half_pixel_centers'],
                              name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_resize_resize(self, params, ie_device, precision, temp_dir):
        if not params['align_corners'] and params['half_pixel_centers'] and params['op_name'] == ['RESIZE_NEAREST_NEIGHBOR']:
            pytest.xfail("CVS-110473")
        self._test(ie_device, precision, temp_dir, params)
