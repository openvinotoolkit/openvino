"""
 Copyright (c) 2019 Intel Corporation

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

from mo.front.kaldi.extractors.batchnorm_component_ext import BatchNormComponentFrontExtractor
from mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading
from mo.ops.op import Op
from mo.ops.scale_shift import ScaleShiftOp


class BatchNormComponentFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['ScaleShift'] = ScaleShiftOp

    @classmethod
    def create_pb_for_test_node(cls):
        pb = KaldiFrontExtractorTest.write_tag_with_value('<Dim>', 16)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<BlockDim>', 16)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<Epsilon>', 0.00001, np.float32)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<TargetRms>', 0.5, np.float32)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<TestMode>', 'F', np.string_)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<Count>', 16)
        pb += b'<StatsMean> '
        pb += KaldiFrontExtractorTest.generate_vector(16)
        pb += b'<StatsVar> '
        pb += KaldiFrontExtractorTest.generate_vector(16)
        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)

    def test_extract(self):
        BatchNormComponentFrontExtractor.extract(self.test_node)
        self.assertEqual(len(self.test_node['embedded_inputs']), 2)
        ref_weights = list([1.5811389e+02, 4.9999750e-01, 3.5355249e-01, 2.8867465e-01, 2.4999970e-01,
                            2.2360659e-01, 2.0412397e-01, 1.8898210e-01, 1.7677659e-01, 1.6666657e-01,
                            1.5811381e-01, 1.5075560e-01, 1.4433751e-01, 1.3867499e-01, 1.3363057e-01, 1.2909940e-01])
        ref_biases = list([-0., -0.4999975, -0.707105, -0.86602396, -0.9999988, -1.1180329,
                           -1.2247438, -1.3228748, -1.4142127, -1.4999992, -1.5811381, -1.6583116,
                           -1.7320502, -1.8027749, -1.870828, -1.936491])
        self.assertEqual(np.allclose(self.test_node['weights'], ref_weights), True)
        self.assertEqual(np.allclose(self.test_node['biases'], ref_biases), True)
