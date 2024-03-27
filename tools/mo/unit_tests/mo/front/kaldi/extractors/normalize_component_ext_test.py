# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.normalize import NormalizeOp
from openvino.tools.mo.front.kaldi.extractors.normalize_component_ext import NormalizeComponentFrontExtractor
from openvino.tools.mo.ops.op import Op
from unit_tests.mo.front.kaldi.extractors.common_ext_test import KaldiFrontExtractorTest
from unit_tests.mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading


class NormalizeComponentFrontExtractorTest(KaldiFrontExtractorTest):
    @classmethod
    def register_op(cls):
        Op.registered_ops['Normalize'] = NormalizeOp

    @classmethod
    def create_pb_for_test_node(cls):
        pb = KaldiFrontExtractorTest.write_tag_with_value('<InputDim>', 16)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<TargetRms>', 0.5, np.float32)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<AddLogStddev>', 'F', np.string_)
        cls.test_node['parameters'] = TestKaldiUtilsLoading.bytesio_from(pb)

    def test_extract(self):
        NormalizeComponentFrontExtractor.extract(self.test_node)
        self.assertEqual(len(self.test_node['embedded_inputs']), 1)
        self.assertListEqual(list(self.test_node['weights']), [2.0])
