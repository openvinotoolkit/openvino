# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.front.tf.next_iteration_ext import NextIterationExtractor
from mo.utils.unittest.extractors import PB, BaseExtractorsTestingClass


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
