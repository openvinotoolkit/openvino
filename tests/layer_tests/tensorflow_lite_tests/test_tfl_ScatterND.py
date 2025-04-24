import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

np.random.seed(42)

test_params = [
    {'indices_shape': [4, 1], 'indices_value': [4, 3, 1, 7], 'updates_dtype': np.int32, 'updates_shape': [4],
     'shape_shape': [1], 'shape_value': [8]},
    {'indices_shape': [4, 1], 'indices_value': [4, 3, 1, 7], 'updates_dtype': np.int64, 'updates_shape': [4],
     'shape_shape': [1], 'shape_value': [8]},
    {'indices_shape': [4, 1], 'indices_value': [4, 3, 1, 7], 'updates_dtype': np.float32, 'updates_shape': [4],
     'shape_shape': [1], 'shape_value': [8]},
    {'indices_shape': [4, 1], 'indices_value': [4, 3, 1, 7], 'updates_dtype': bool, 'updates_shape': [4],
     'shape_shape': [1], 'shape_value': [8]},

    {'indices_shape': [4, 2], 'indices_value': [[0, 0], [1, 0], [0, 2], [1, 2]], 'updates_dtype': np.int32,
     'updates_shape': [4, 5], 'shape_shape': [3], 'shape_value': [2, 3, 5]},
    {'indices_shape': [4, 2], 'indices_value': [[0, 0], [1, 0], [0, 2], [1, 2]], 'updates_dtype': np.int64,
     'updates_shape': [4, 5], 'shape_shape': [3], 'shape_value': [2, 3, 5]},
    {'indices_shape': [4, 2], 'indices_value': [[0, 0], [1, 0], [0, 2], [1, 2]], 'updates_dtype': np.float32,
     'updates_shape': [4, 5], 'shape_shape': [3], 'shape_value': [2, 3, 5]},
    {'indices_shape': [4, 2], 'indices_value': [[0, 0], [1, 0], [0, 2], [1, 2]], 'updates_dtype': bool,
     'updates_shape': [4, 5], 'shape_shape': [3], 'shape_value': [2, 3, 5]},
]


class TestTFLiteScatterNDLayerTest(TFLiteLayerTest):
    inputs = ["Indices", "Updates", "Shape"]
    outputs = ["ScatterND"]
    allowed_ops = ['SCATTER_ND']

    def _prepare_input(self, inputs_dict, generator=None):
        inputs_dict['Indices'] = np.array(self.indices_value, dtype=np.int32).reshape(self.indices_shape)

        if self.updates_dtype in [tf.int32, tf.int64]:
            inputs_dict['Updates'] = np.random.randint(-100, 100 + 1, self.updates_shape)
        if self.updates_dtype == tf.float32:
            inputs_dict['Updates'] = (100 - (-100) * np.random.random_sample(self.updates_shape) + (-100))
        if self.updates_dtype == tf.bool:
            inputs_dict['Updates'] = np.random.choice([True, False], size=self.updates_shape)

        inputs_dict['Updates'] = inputs_dict['Updates'].astype(self.updates_dtype)
        inputs_dict['Shape'] = np.array(self.shape_value, dtype=np.int32)

        return inputs_dict

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'indices_shape', 'indices_value',
                                                    'updates_dtype', 'updates_shape', 'shape_shape',
                                                    'shape_value'})) == 6, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.indices_value = params['indices_value']
        self.indices_shape = params['indices_shape']
        self.updates_dtype = params['updates_dtype']
        self.updates_shape = params['updates_shape']
        self.shape_value = params['shape_value']

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            indices = tf.compat.v1.placeholder(tf.int32, self.indices_shape, name=self.inputs[0])
            updates = tf.compat.v1.placeholder(self.updates_dtype, self.updates_shape, name=self.inputs[1])
            shape = tf.compat.v1.placeholder(tf.int32, params['shape_shape'], name=self.inputs[2])

            tf.scatter_nd(indices, updates, shape, name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_scatter_nd(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
