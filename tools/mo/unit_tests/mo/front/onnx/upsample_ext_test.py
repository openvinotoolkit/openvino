# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import onnx

from openvino.tools.mo.front.onnx.upsample_ext import UpsampleFrontExtractor
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.extractors import BaseExtractorsTestingClass
from unit_tests.utils.graph import build_graph


class UpsampleONNXExtractorTest(BaseExtractorsTestingClass):
    @staticmethod
    def _create_node(attrs: dict):
        pb = onnx.helper.make_node("Upsample", ["X"], ["Y"], **attrs)
        graph = build_graph({'node_0': {'pb': pb}}, [])
        return Node(graph, 'node_0')

    @staticmethod
    def _base_attrs():
        # Commonly used attributes in the tests
        # Each test takes these ones and then adds/modifies/deletes particular fields
        return (
            # test input ONNX attributes
            dict(
                mode='nearest',
                scales=[1., 1., 2., 2.],
            ),
            # reference output Node attributes
            dict(
                width_scale=2.0,
                height_scale=2.0,
                mode='nearest',
            )
        )

    @staticmethod
    def _extract(inp):
        node = __class__._create_node(inp)
        UpsampleFrontExtractor.extract(node)
        return node

    def _match(self, out, ref):
        self.res = out
        self.expected = ref
        self.compare()

    def test_all_valid_default(self):
        inp, ref = self._base_attrs()
        out = self._extract(inp)
        self._match(out, ref)

    def test_invalid_mode(self):
        inp, ref = self._base_attrs()
        inp['mode'] = 'invalid_mode'
        with self.assertRaisesRegex(Error, '.*decoding Upsample.*supported modes.*'):
            out = self._extract(inp)

    def test_invalid_scales(self):
        inp, ref = self._base_attrs()
        inp['scales'] = [1.5, 1.5, 2.0, 2.0]
        with self.assertRaisesRegex(Error, '.*Upsampling of batch and feature dimensions is not supported for node.*'):
            out = self._extract(inp)

    def test_invalid_2D_scales(self):
        inp, ref = self._base_attrs()
        inp['scales'] = [2.0, 2.0]
        with self.assertRaisesRegex(Error,
                                    '.*Upsample scales attribute is wrong for node.*. Only 4D scales are supported.'):
            out = self._extract(inp)
