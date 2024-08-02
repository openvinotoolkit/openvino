# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import onnx

from openvino.tools.mo.front.onnx.instance_normalization_ext import InstanceNormalizationExtractor
from unit_tests.utils.extractors import PB, BaseExtractorsTestingClass


class TestInstanceNormalization(BaseExtractorsTestingClass):
    @staticmethod
    def _create_node():
        pb = onnx.helper.make_node(
            'InstanceNormalization',
            inputs=['a'],
            outputs=['b'],
            epsilon=0.5,
        )
        node = PB({'pb': pb})
        return node

    def test_image_scaler_ext(self):
        node = self._create_node()
        InstanceNormalizationExtractor.extract(node)
        self.res = node

        self.expected = {
            'epsilon': 0.5,
        }

        self.compare()
