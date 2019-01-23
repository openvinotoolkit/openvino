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

from mo.front.tf.extractor import tf_op_extractors
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass

dtypes_map = [(1, np.float32), (2, np.float64), (3, np.int32), (4, np.uint8), (5, np.int16), (6, np.int8),
              (7, str), (9, np.int64), (10, bool), (17, np.uint16)]

if tf.__version__ > "1.4.0":
    dtypes_map.extend([(22, np.uint32), (23, np.uint64)])


@generator
class EltwiseExtractorTest(BaseExtractorsTestingClass):
    def check_lambda_res(self, actual, expected, expected_type):
        self.assertEqual(expected_type, type(actual), "Eltwise lambda function results has wrong data type!")
        if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
            np.testing.assert_equal(actual, expected)
        else:
            self.assertEqual(expected, actual, "Eltwise lambda function results validation failed!")

    @classmethod
    def setUpClass(cls):
        cls.patcher = 'mo.front.tf.extractors.eltwise.eltwise_infer'

    @generate(*dtypes_map)
    def test_eltwise_dtypes_map(self, dtype, np_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": dtype
                    })
                })
            })
        })
        self.expected = {
            'can_be_bias': True,
            'data_type': np_type,
            'operation': 'sum',
            'type': 'Eltwise'
        }
        self.res = tf_op_extractors['Add'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = None
        self.compare()

    @generate(*[((1, 2), 3, int), ((1., 2.), 3., float), (('a', 'b'), 'ab', str),
                ((np.full(shape=(10, 10), fill_value=4), np.full(shape=(10, 10), fill_value=2)),
                 np.full(shape=(10, 10), fill_value=6), np.ndarray)
                ])
    def test_eltwise_add(self, lambda_args, expected_res, expected_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'can_be_bias': True,
            'data_type': np.float32,
            'operation': 'sum',
            'type': 'Eltwise'
        }
        self.res = tf_op_extractors['Add'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        eltwise_lambda = self.call_args[0][1]
        lambda_res = eltwise_lambda(*lambda_args)
        self.check_lambda_res(actual=lambda_res, expected=expected_res, expected_type=expected_type)
        self.expected_call_args = None
        self.compare()

    @generate(*[((1, 2), 2, int), ((1., 2.), 2., float),
                ((np.full(shape=(10, 10), fill_value=4), np.full(shape=(10, 10), fill_value=2)),
                 np.full(shape=(10, 10), fill_value=8), np.ndarray)])
    def test_eltwise_mul(self, lambda_args, expected_res, expected_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'data_type': np.float32,
            'operation': 'mul',
            'type': 'Eltwise'
        }
        self.res = tf_op_extractors['Mul'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        eltwise_lambda = self.call_args[0][1]
        lambda_res = eltwise_lambda(*lambda_args)
        self.check_lambda_res(actual=lambda_res, expected=expected_res, expected_type=expected_type)
        self.expected_call_args = None
        self.compare()

    @generate(*[(4, 0.5, np.float64),
                (np.full(shape=(10, 10), fill_value=4), np.full(shape=(10, 10), fill_value=0.5), np.ndarray)])
    def test_eltwise_rsqrt(self, lambda_args, expected_res, expected_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'data_type': np.float32,
            'type': 'Power',
            'power': -0.5,
            'scale': 1,
            'shift': 0
        }
        self.res = tf_op_extractors['Rsqrt'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        eltwise_lambda = self.call_args[0][1]
        lambda_res = eltwise_lambda(lambda_args)
        self.check_lambda_res(actual=lambda_res, expected=expected_res, expected_type=expected_type)
        self.expected_call_args = None
        self.compare()

    @generate(*[(1, -1, int), (1., -1., float),
                (np.full(shape=(10, 10), fill_value=4), np.full(shape=(10, 10), fill_value=-4), np.ndarray)])
    def test_eltwise_neg(self, lambda_args, expected_res, expected_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'data_type': np.float32,
            'type': 'Power',
            'power': 1,
            'scale': -1,
            'shift': 0
        }
        self.res = tf_op_extractors['Neg'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        eltwise_lambda = self.call_args[0][1]
        lambda_res = eltwise_lambda(lambda_args)
        self.check_lambda_res(actual=lambda_res, expected=expected_res, expected_type=expected_type)
        self.expected_call_args = None
        self.compare()

    @generate(*[((1, 2), -1, int), ((1., 2.), -1., float),
                ((np.full(shape=(10, 10), fill_value=4), np.full(shape=(10, 10), fill_value=2)),
                 np.full(shape=(10, 10), fill_value=2), np.ndarray)
                ])
    def test_eltwise_sub(self, lambda_args, expected_res, expected_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'data_type': np.float32,
        }
        self.res = tf_op_extractors['Sub'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        eltwise_lambda = self.call_args[0][1]
        lambda_res = eltwise_lambda(*lambda_args)
        self.check_lambda_res(actual=lambda_res, expected=expected_res, expected_type=expected_type)
        self.expected_call_args = None
        self.compare()

    @generate(*[(4, 4, np.int64), (4., 4., np.float64),
                (-1, 0, np.int64), (-1., 0, np.float64),
                (np.full(shape=(3, 3), fill_value=-1), np.zeros(shape=(3, 3)), np.ndarray),
                (np.full(shape=(3, 3), fill_value=4), np.full(shape=(3, 3), fill_value=4), np.ndarray)])
    def test_eltwise_relu(self, lambda_args, expected_res, expected_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'data_type': np.float32,
            "type": "ReLU"
        }
        self.res = tf_op_extractors['Relu'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        eltwise_lambda = self.call_args[0][1]
        lambda_res = eltwise_lambda(lambda_args)
        self.check_lambda_res(actual=lambda_res, expected=expected_res, expected_type=expected_type)
        self.expected_call_args = None
        self.compare()

    @generate(*[(4, 4, np.int64), (4., 4., np.float64),
                (-1, 0, np.int64), (-1., 0, np.float64),
                (10, 6, np.int64), (10., 6, np.float64),
                (np.full(shape=(3, 3), fill_value=-1), np.zeros(shape=(3, 3)), np.ndarray),
                (np.full(shape=(3, 3), fill_value=10), np.full(shape=(3, 3), fill_value=6), np.ndarray),
                (np.full(shape=(3, 3), fill_value=4), np.full(shape=(3, 3), fill_value=4), np.ndarray)])
    def test_eltwise_relu6(self, lambda_args, expected_res, expected_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'data_type': np.float32,
            'type': 'Clamp',
            'min': 0,
            'max': 6
        }
        self.res = tf_op_extractors['Relu6'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        eltwise_lambda = self.call_args[0][1]
        lambda_res = eltwise_lambda(lambda_args)
        self.check_lambda_res(actual=lambda_res, expected=expected_res, expected_type=expected_type)
        self.expected_call_args = None
        self.compare()

    @generate(*[(4, 16, int), (4., 16., float),
                (-1, 1, int), (-1., 1., float),
                (np.full(shape=(3, 3), fill_value=-1), np.full(shape=(3, 3), fill_value=1), np.ndarray),
                (np.full(shape=(3, 3), fill_value=4), np.full(shape=(3, 3), fill_value=16), np.ndarray)])
    def test_eltwise_square(self, lambda_args, expected_res, expected_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'data_type': np.float32,
        }
        self.res = tf_op_extractors['Square'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        eltwise_lambda = self.call_args[0][1]
        lambda_res = eltwise_lambda(lambda_args)
        self.check_lambda_res(actual=lambda_res, expected=expected_res, expected_type=expected_type)
        self.expected_call_args = None
        self.compare()

    @generate(*[
        ((4, 16), 16, np.int64),
        ((4., 16.), 16., np.float64),
        ((-1, 1), 1, np.int64),
        ((-1., 1), 1., np.float64),
        (
                (
                        np.full(shape=(3, 3), fill_value=-1),
                        np.full(shape=(3, 3), fill_value=-2)
                ),
                np.full(shape=(3, 3), fill_value=-1),
                np.ndarray
        ),
        (
                (
                        np.full(shape=(3, 3), fill_value=4),
                        np.full(shape=(3, 3), fill_value=0)
                ),
                np.full(shape=(3, 3), fill_value=4),
                np.ndarray)
    ])
    def test_eltwise_maximum(self, lambda_args, expected_res, expected_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'data_type': np.float32,
        }
        self.res = tf_op_extractors['Maximum'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        eltwise_lambda = self.call_args[0][1]
        lambda_res = eltwise_lambda(*lambda_args)
        self.check_lambda_res(actual=lambda_res, expected=expected_res, expected_type=expected_type)
        self.expected_call_args = None
        self.compare()

    @generate(*[((16, 4), 4, float), ((4., 16.), 0.25, float),
                ((-1, 1), -1, float), ((-16., 1), -16., float),
                ((np.full(shape=(3, 3), fill_value=-1), np.full(shape=(3, 3), fill_value=-2)),
                 np.full(shape=(3, 3), fill_value=0.5), np.ndarray),
                ((np.full(shape=(3, 3), fill_value=4), np.full(shape=(3, 3), fill_value=0)),
                 np.full(shape=(3, 3), fill_value=np.inf), np.ndarray)])
    def test_eltwise_realdiv(self, lambda_args, expected_res, expected_type):
        node_pb = PB({
            'pb': PB({
                'attr': PB({
                    'T': PB({
                        "type": 1
                    })
                })
            })
        })
        self.expected = {
            'data_type': np.float32,
        }
        self.res = tf_op_extractors['RealDiv'](node_pb)
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        eltwise_lambda = self.call_args[0][1]
        lambda_res = eltwise_lambda(*lambda_args)
        self.check_lambda_res(actual=lambda_res, expected=expected_res, expected_type=expected_type)
        self.expected_call_args = None
        self.compare()
