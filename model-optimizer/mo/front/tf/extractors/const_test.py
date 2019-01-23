"""
 Copyright (c) 2018 Intel Corporation

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

import numpy as np
import tensorflow as tf
from generator import generator, generate

from mo.front.tf.extractors.const import tf_const_ext
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass

dtypes = {"ints": [(tf.int8, np.int8),
                   (tf.int16, np.int16),
                   (tf.int32, np.int32),
                   (tf.int64, np.int64)],

          "uints": [(tf.uint8, np.uint8),
                    (tf.uint16, np.uint16),
                    ],
          "floats": [(tf.float32, np.float32),
                     (tf.float64, np.double)],

          "bools": [(tf.bool, np.bool)],

          "strings": [(tf.string, np.str)]}
if tf.__version__ > "1.4.0":
    dtypes['uints'].extend([(tf.uint32, np.uint32), (tf.uint64, np.uint64)])


@generator
class ConstExtractorTest(BaseExtractorsTestingClass):
    @classmethod
    def setUpClass(cls):
        cls.patcher = 'mo.front.tf.extractors.const.tf_const_infer'

    @generate(*dtypes['ints'])
    def test_const_ints(self, tf_dtype, np_dtype):
        shape = [1, 1, 200, 50]
        values = np.random.randint(low=np.iinfo(np_dtype).min, high=np.iinfo(np_dtype).max, size=shape, dtype=np_dtype)
        tensor_proto = tf.make_tensor_proto(values=values, dtype=tf_dtype, shape=shape)
        pb = PB({"attr": PB({
            "value": PB({
                "tensor": PB({
                    "dtype": tensor_proto.dtype,
                    "tensor_shape": tensor_proto.tensor_shape,
                    "tensor_content": tensor_proto.tensor_content
                })
            })
        })})
        self.expected = {
            'data_type': np_dtype,
            'shape': np.asarray(shape, dtype=np.int),
            'value': values
        }
        self.res = tf_const_ext(pb=pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = None
        self.compare()

    @generate(*dtypes['uints'])
    def test_const_uints(self, tf_dtype, np_dtype):
        shape = [1, 1, 200, 50]
        values = np.random.randint(low=np.iinfo(np_dtype).min, high=np.iinfo(np_dtype).max, size=shape, dtype=np_dtype)
        tensor_proto = tf.make_tensor_proto(values=values, dtype=tf_dtype, shape=shape)
        pb = PB({"attr": PB({
            "value": PB({
                "tensor": PB({
                    "dtype": tensor_proto.dtype,
                    "tensor_shape": tensor_proto.tensor_shape,
                })
            })
        })})
        if tf_dtype == tf.uint16:
            setattr(pb.attr.value.tensor, "int_val", values.tolist())
        else:
            setattr(pb.attr.value.tensor, "tensor_content", tensor_proto.tensor_content)
        self.expected = {
            'data_type': np_dtype,
            'shape': np.asarray(shape, dtype=np.int),
            'value': values
        }
        self.res = tf_const_ext(pb=pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = None
        self.compare()

    @generate(*dtypes['floats'])
    def test_const_floats(self, tf_dtype, np_dtype):
        shape = [1, 1, 200, 50]
        values = np.random.uniform(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max, size=shape).astype(
            np_dtype)
        tensor_proto = tf.make_tensor_proto(values=values, dtype=tf_dtype, shape=shape)
        pb = PB({"attr": PB({
            "value": PB({
                "tensor": PB({
                    "dtype": tensor_proto.dtype,
                    "tensor_shape": tensor_proto.tensor_shape,
                    "tensor_content": tensor_proto.tensor_content
                })
            })
        })})
        self.expected = {
            'data_type': np_dtype,
            'shape': np.asarray(shape, dtype=np.int),
            'value': values
        }
        self.res = tf_const_ext(pb=pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = None
        self.compare()

    # TODO: Check how to correctly handle tensor_proto with booleans. It has no tensor_content section
    @generate(*dtypes['bools'])
    def test_const_floats(self, tf_dtype, np_dtype):
        shape = [1, 1, 50, 50]
        values = np.random.choice(a=[True, False], size=shape, p=[0.5, 0.5])
        tensor_proto = tf.make_tensor_proto(values=values, dtype=tf_dtype, shape=shape)
        pb = PB({"attr": PB({
            "value": PB({
                "tensor": PB({
                    "dtype": tensor_proto.dtype,
                    "tensor_shape": tensor_proto.tensor_shape,
                    "bool_val": values.tolist()
                })
            })
        })})
        self.expected = {
            'data_type': np_dtype,
            'shape': np.asarray(shape, dtype=np.int),
            'value': values
        }
        self.res = tf_const_ext(pb=pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = None
        self.compare()
        # TODO: Check how to correctly create tensor_proto with strings
        # @generate(*dtypes['strings'])
        # def test_const_floats(self, tf_dtype, np_dtype):
        #     shape = [1, 1, 50, 50]
        #     values = np.chararray(shape=shape)
        #     values[:] = "bla"
        #     tensor_proto = tf.make_tensor_proto(values=values, dtype=tf_dtype, shape=shape)
        #     pb = PB({"attr": PB({
        #         "value": PB({
        #             "tensor": PB({
        #                 "dtype": tensor_proto.dtype,
        #                 "tensor_shape": tensor_proto.tensor_shape,
        #                 "tensor_content": tensor_proto.tensor_content
        #             })
        #         })
        #     })})
        #     self.expected = {
        #         'data_type': np_dtype,
        #         'shape': np.asarray(shape, dtype=np.int),
        #         'value': values
        #     }
        #     self.res = tf_const_ext(pb=pb)
        #     self.res["infer"](None)
        #     self.call_args = self.infer_mock.call_args
        #     self.expected_call_args = None
        #     self.compare()
