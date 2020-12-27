"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import unittest

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.extractors.utils import collect_tf_attrs, tf_tensor_content
from mo.utils.unittest.extractors import PB


class AttrParsingTest(unittest.TestCase):
    def test_simple_check(self):
        pb = PB({'attr': {
            'str': PB({'s': "aaaa"}),
            'int': PB({'i': 7}),
            'float': PB({'f': 2.0}),
            'bool': PB({'b': True}),
            'lisint': PB({'list': PB({'i': 5, 'i': 6})})}
        })

        res = collect_tf_attrs(pb.attr)

        # Reference results for given parameters
        ref = {
            'str': pb.attr['str'].s,
            'int': pb.attr['int'].i,
            'float': pb.attr['float'].f,
            'bool': pb.attr['bool'].b,
            'lisint': pb.attr['lisint'].list.i
        }
        for attr in ref:
            self.assertEqual(res[attr], ref[attr])


class TensorContentParsing(unittest.TestCase):
    def test_list_not_type_no_shape(self):
        pb_tensor = PB(dict(
            dtype=3,
            tensor_content=b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00'
        ))
        tf_dtype = pb_tensor.dtype
        shape = np.array([5])
        ref = [1, 2, 3, 4, 5]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_list_type_no_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'int_val': np.array([1, 2, 3, 4, 5], dtype=np.int32)
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([5])
        ref = [1, 2, 3, 4, 5]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_list_not_type_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'tensor_content': b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00'
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([10])
        ref = [1, 2, 3, 4, 5, 5, 5, 5, 5, 5]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_list_type_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'int_val': np.array([1, 2, 3, 4, 5], dtype=np.int32)
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([10])
        ref = [1, 2, 3, 4, 5, 5, 5, 5, 5, 5]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_0d_not_type_no_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'tensor_content': b'\x01\x00\x00\x00',
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([])
        ref = 1
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(res == ref)

    def test_0d_type_no_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'int_val': [5],
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([])
        ref = 5
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(res == ref)

    def test_0d_not_type_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'tensor_content': b'\x01\x00\x00\x00',
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([3])
        ref = [1, 1, 1]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_0d_type_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'int_val': [5],
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([3])
        ref = [5, 5, 5]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_0d_type_shape_1(self):
        pb_tensor = PB({
            'dtype': 3,
            'int_val': [5],
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([1])
        ref = [5]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_nd_not_type_no_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'tensor_content':
                b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00',
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([2, 3])
        ref = [[1, 2, 3], [4, 5, 6]]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_nd_type_no_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'int_val': [[1, 2, 3], [4, 5, 6]],
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([2, 3])
        ref = [[1, 2, 3], [4, 5, 6]]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_nd_not_type_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'tensor_content':
                b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00',
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([2, 5, 2])
        ref = [[[1, 2], [3, 4], [5, 6], [6, 6], [6, 6]],
               [[6, 6], [6, 6], [6, 6], [6, 6], [6, 6]]]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_nd_type_shape(self):
        pb_tensor = PB({
            'dtype': 3,
            'int_val': [[1, 2, 3], [4, 5, 6]],
        })
        tf_dtype = pb_tensor.dtype
        shape = np.array([2, 5, 2])
        ref = [[[1, 2], [3, 4], [5, 6], [6, 6], [6, 6]],
               [[6, 6], [6, 6], [6, 6], [6, 6], [6, 6]]]
        res = tf_tensor_content(tf_dtype, shape, pb_tensor)
        self.assertTrue(np.all(res == ref))

    def test_str_decode(self):
        pb_tensor = PB({
            'dtype': 7,
            'string_val': [b"\037\000\036\000\002\000\303\237\035\000\002"]
        })
        tf_dtype = pb_tensor.dtype
        shape = int64_array([1])
        warning_message = 'ERROR:root:Failed to parse a tensor with Unicode characters. Note that Inference Engine ' \
                          'does not support string literals, so the string constant should be eliminated from the ' \
                          'graph.'
        ref_val = np.array([b'\x1f\x00\x1e\x00\x02\x00\xc3\x9f\x1d\x00\x02'])
        with self.assertLogs(log.getLogger(), level="ERROR") as cm:
            result = tf_tensor_content(tf_dtype, shape, pb_tensor)
            self.assertEqual([warning_message], cm.output)
            self.assertEqual(ref_val, result)

