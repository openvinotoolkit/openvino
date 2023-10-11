import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest

test_params = [
    {'shape': [12, 2, 2, 5], 'dtype': np.int32, 'strides': [2, 1, 3, 1], 'begin': [0, 0, 0, 0], 'end': [12, 2, 2, 5],
     'begin_mask': None, 'end_mask': None, 'shrink_axis_mask': 4},
    {'shape': [12, 2, 2, 5], 'dtype': np.float32, 'strides': None, 'begin': [0, 0, 0, 0], 'end': [12, 2, 2, 5],
     'begin_mask': None, 'end_mask': None, 'shrink_axis_mask': None},
    {'shape': [12, 2, 2, 5], 'dtype': np.float32, 'strides': None, 'begin': [1, 0, 1, 0], 'end': [8, 2, 2, 3],
     'begin_mask': 8, 'end_mask': 3, 'shrink_axis_mask': 4},
    {'shape': [12, 2, 2, 5], 'dtype': np.int64, 'strides': [1], 'begin': [0], 'end': [1],
     'begin_mask': 8, 'end_mask': 3, 'shrink_axis_mask': None},
    {'shape': [12, 2, 2, 5], 'dtype': bool, 'strides': [1], 'begin': [0], 'end': [1],
     'begin_mask': 8, 'end_mask': 3, 'shrink_axis_mask': None},
]


class TestTFLiteStridedSliceLayerTest(TFLiteLayerTest):
    inputs = ["Input", "Begin", "End"]
    outputs = ["StridedSlice"]
    allowed_ops = ['STRIDED_SLICE']

    def _prepare_input(self, inputs_dict, generator=None):
        if self.input_dtype == bool:
            inputs_dict['Input'] = np.random.choice([True, False], size=inputs_dict['Input'])
        else:
            inputs_dict['Input'] = np.random.randint(-255, 255, inputs_dict['Input']).astype(self.input_dtype)
        inputs_dict['Begin'] = np.array(self.begin).astype(np.int32)
        inputs_dict['End'] = np.array(self.end).astype(np.int32)
        if self.strides:
            inputs_dict["Strides"] = np.array(self.strides).astype(np.int32)

        return inputs_dict

    def make_model(self, params):
        assert len(set(params.keys()).intersection({'shape', 'dtype', 'strides', 'begin',
                                                    'end', 'begin_mask', 'end_mask', 'shrink_axis_mask'})) == 8, \
            'Unexpected parameters for test: ' + ','.join(params.keys())
        self.input_dtype = params['dtype']
        self.begin = params['begin']
        self.end = params['end']
        self.strides = params['strides']
        if "Strides" in self.inputs:
            self.inputs.pop(-1)
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            placeholder = tf.compat.v1.placeholder(params['dtype'], params['shape'], self.inputs[0])
            begin = tf.compat.v1.placeholder(tf.int32, [len(params['begin'])], self.inputs[1])
            end = tf.compat.v1.placeholder(tf.int32, [len(params['end'])], self.inputs[2])
            strides = None
            if params['strides']:
                name = "Strides"
                self.inputs.append(name)
                strides = tf.compat.v1.placeholder(tf.int32, [len(params['end'])], name)

            tf.strided_slice(placeholder, begin, end, strides,
                             begin_mask=params['begin_mask'], end_mask=params['end_mask'],
                             shrink_axis_mask=params['shrink_axis_mask'], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_strided_slice(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, params)
