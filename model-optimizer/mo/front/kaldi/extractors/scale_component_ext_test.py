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

from mo.front.kaldi.extractors.scale_component_ext import FixedScaleComponentFrontExtractor
from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading
from mo.ops.op import Op
from mo.ops.scale_shift import ScaleShiftOp


class FixedScaleComponentFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['ScaleShift'] = ScaleShiftOp

    @classmethod
    def create_pb_for_test_node(cls):
        cls.input_shape = 10

        pb = b'<FixedScaleComponent> <Scales> '
        pb += KaldiFrontExtractorTest.generate_vector(cls.input_shape)
        pb += b'</FixedScaleComponent>'

        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)
        FixedScaleComponentFrontExtractor.extract(cls.test_node)

    def test_fixedscale_extractor(self):
        input_shape = FixedScaleComponentFrontExtractorTest.input_shape

        exp_res = {
            'op': 'ScaleShift',
            'layout': 'NCHW',
            'out-size': input_shape,
            'weights': np.arange(input_shape)
        }

        self.compare_node_attrs(exp_res)
