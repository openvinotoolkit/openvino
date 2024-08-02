# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.tf.next_iteration_ext import NextIterationExtractor
from unit_tests.utils.extractors import PB, BaseExtractorsTestingClass


class TestNextIteration(BaseExtractorsTestingClass):
    def test_is_cyclic(self):
        pb = PB({})
        node = PB({'pb': pb})
        NextIterationExtractor.extract(node)
        self.expected = {
            'is_cyclic': True,
        }
        self.res = node
        self.compare()
