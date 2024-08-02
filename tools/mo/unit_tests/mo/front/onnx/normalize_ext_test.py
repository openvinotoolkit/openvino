# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import onnx

from openvino.tools.mo.front.onnx.normalize_ext import NormalizeFrontExtractor
from unit_tests.utils.extractors import PB, BaseExtractorsTestingClass


class TestNormalize(BaseExtractorsTestingClass):
    @staticmethod
    def _create_node(across_spatial=None, channel_shared=None, eps=None):
        if across_spatial is None:
            across_spatial = 0
        if channel_shared is None:
            channel_shared = 0
        if eps is None:
            eps = 0.1
        pb = onnx.helper.make_node(
            'Normalize',
            across_spatial=across_spatial,
            channel_shared=channel_shared,
            eps=eps,
            inputs=['a'],
            outputs=['b']
        )
        node = PB({'pb': pb})
        return node

    def test_ok(self):
        node = self._create_node()
        NormalizeFrontExtractor.extract(node)
        self.res = node

        self.expected = {
            'across_spatial': False,
            'channel_shared': False,
            'eps': 0.1
        }

        self.compare()
