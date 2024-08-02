# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import onnx

from openvino.tools.mo.front.onnx.pad_ext import PadFrontExtractor
from openvino.tools.mo.graph.graph import Graph
from unit_tests.utils.extractors import PB, BaseExtractorsTestingClass


class TestPad(BaseExtractorsTestingClass):
    @staticmethod
    def _create_node(pads=None, value=None, mode=None):
        if pads is None:
            pads = [1, 2, 3, 4]
        if value is None:
            value = 0.0
        if mode is None:
            mode = 'constant'
        pb = onnx.helper.make_node(
            'Pad',
            pads=pads,
            mode=mode,
            value=value,
            inputs=['a'],
            outputs=['b']
        )
        graph = Graph()
        node = PB({'pb': pb, 'graph': graph})

        return node

    def test_ok(self):
        node = self._create_node()
        PadFrontExtractor.extract(node)
        self.res = node

        self.expected = {
            'pads': [[1, 3], [2, 4]],
            'mode': 'constant',
            'fill_value': 0
        }

        self.compare()

    def test_older_pad_opset_11(self):
        node = self._create_node()
        node.graph.graph['fw_opset_version'] = 11
        PadFrontExtractor.extract(node)
        self.res = node

        self.expected = {
            'pads': [[1, 3], [2, 4]],
            'mode': 'constant',
            'fill_value': 0
        }

        self.compare()

    def test_reflect(self):
        node = self._create_node(mode='reflect')
        PadFrontExtractor.extract(node)
        self.res = node

        self.expected = {
            'pads': [[1, 3], [2, 4]],
            'mode': 'reflect',
            'fill_value': 0
        }

        self.compare()

    def test_non_zero_fill_value(self):
        node = self._create_node(value=1.0)
        PadFrontExtractor.extract(node)
        self.res = node

        self.expected = {
            'pads': [[1, 3], [2, 4]],
            'mode': 'constant',
            'fill_value': 1.0
        }

        self.compare()
