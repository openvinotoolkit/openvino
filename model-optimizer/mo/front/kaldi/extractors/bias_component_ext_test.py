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

import numpy as np

from mo.front.kaldi.extractors.bias_component_ext import FixedBiasComponentFrontExtractor
from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading
from mo.ops.op import Op
from mo.ops.scale_shift import ScaleShiftOp


class FixedBiasComponentFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['ScaleShift'] = ScaleShiftOp

    @classmethod
    def create_pb_for_test_node(cls):
        cls.input_shape = 10

        pb = b'<FixedBiasComponent> <Bias> '
        pb += KaldiFrontExtractorTest.generate_vector(cls.input_shape)
        pb += b'</FixedBiasComponent>'

        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)
        FixedBiasComponentFrontExtractor.extract(cls.test_node)

    def test_fixedbias_extractor(self):
        input_shape = FixedBiasComponentFrontExtractorTest.input_shape

        exp_res = {
            'op': 'ScaleShift',
            'layout': 'NCHW',
            'bias_term': True,
            'out-size': input_shape,
            'biases': np.arange(input_shape)
        }

        self.compare_node_attrs(exp_res)
