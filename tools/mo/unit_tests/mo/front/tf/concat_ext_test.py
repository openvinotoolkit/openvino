# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.tf.concat_ext import ConcatFrontExtractor
from unit_tests.utils.extractors import PB, BaseExtractorsTestingClass


class ConcatExtractorTest(BaseExtractorsTestingClass):
    def test_concat(self):
        node = PB({'pb': PB({'attr': {'N': PB({'i': 4})}})})
        self.expected = {
            'N': 4,
            'simple_concat': True,
            'type': 'Concat',
            'op': 'Concat',
            'kind': 'op',
            'axis': 1
        }
        ConcatFrontExtractor.extract(node)
        self.res = node
        self.compare()
