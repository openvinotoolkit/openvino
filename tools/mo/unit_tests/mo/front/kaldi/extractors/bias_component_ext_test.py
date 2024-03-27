# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.kaldi.extractors.bias_component_ext import FixedBiasComponentFrontExtractor
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.ops.scale_shift import ScaleShiftOp
from unit_tests.mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from unit_tests.mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading


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
