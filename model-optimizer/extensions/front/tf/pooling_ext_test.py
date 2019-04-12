"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.front.tf.pooling_ext import AvgPoolFrontExtractor, MaxPoolFrontExtractor
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


class PoolingExtractorTest(BaseExtractorsTestingClass):
    @classmethod
    def setUpClass(cls):
        cls.strides = [1, 2, 3, 4]
        cls.ksize = [1, 3, 3, 1]
        cls.patcher = 'mo.ops.pooling.Pooling.infer'

    def test_pool_defaults(self):
        pb = PB({'attr': {
            'data_format': PB({
                's': b"NHWC"
            }),
            'strides': PB({
                'list': PB({
                    "i": self.strides
                })
            }),
            'ksize': PB({
                'list': PB({"i": self.ksize})
            }),
            'padding': PB({
                's': b'VALID'
            })
        }})
        self.expected = {
            'pad': None,  # will be inferred when input shape is known
            'pad_spatial_shape': None,
            'type': 'Pooling',
            'exclude_pad': 'true',
        }
        node = PB({'pb': pb})
        AvgPoolFrontExtractor.extract(node)
        self.res = node
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = (None, None)
        self.compare()

    def test_avg_pool_nhwc(self):
        pb = PB({'attr': {
            'data_format': PB({
                's': b"NHWC"
            }),
            'strides': PB({
                'list': PB({"i": self.strides})
            }),
            'ksize': PB({
                'list': PB({"i": self.ksize})
            }),
            'padding': PB({
                's': b'VALID'
            })
        }})
        self.expected = {
            'window': np.array(self.ksize, dtype=np.int8),
            'spatial_dims': [1, 2],
            'stride': np.array(self.strides, dtype=np.int8),
            'pool_method': "avg",
        }
        node = PB({'pb': pb})
        AvgPoolFrontExtractor.extract(node)
        self.res = node
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = (None, "avg")
        self.compare()

    def test_avg_pool_nchw(self):
        pb = PB({'attr': {
            'data_format': PB({
                's': b"NCHW"
            }),
            'strides': PB({
                'list': PB({
                    "i": self.strides
                })
            }),
            'ksize': PB({
                'list': PB({
                    "i": self.ksize
                })
            }),
            'padding': PB({
                's': b'VALID'
            })
        }})
        self.expected = {
            'window': np.array(self.ksize, dtype=np.int8),
            'spatial_dims': [2, 3],
            'stride': np.array(self.strides, dtype=np.int8),
            'pool_method': "avg",
        }
        node = PB({'pb': pb})
        AvgPoolFrontExtractor.extract(node)
        self.res = node
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = (None, "avg")
        self.compare()

    def test_max_pool_nhwc(self):
        pb = PB({'attr': {
            'data_format': PB({
                's': b"NHWC"
            }),
            'strides': PB({
                'list': PB({
                    "i": self.strides
                })
            }),
            'ksize': PB({
                'list': PB({
                    "i": self.ksize
                })
            }),
            'padding': PB({
                's': b'VALID'
            })
        }})
        self.expected = {
            'window': np.array(self.ksize, dtype=np.int8),
            'spatial_dims': [1, 2],
            'stride': np.array(self.strides, dtype=np.int64),
            'pool_method': "max",
        }
        node = PB({'pb': pb})
        MaxPoolFrontExtractor.extract(node)
        self.res = node
        self.res["infer"](None)
        self.call_args = self.infer_mock.call_args
        self.expected_call_args = (None, "max")
        self.compare()
