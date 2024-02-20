# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.ops.instance_normalization import InstanceNormalization
from openvino.tools.mo.graph.graph import Graph


class InstanceNormalizationOp(unittest.TestCase):
    def test_constructor_supported_attrs(self):
        graph = Graph()
        op = InstanceNormalization(graph, attrs={'epsilon': 0.1})
        self.assertEqual(op.supported_attrs(), ['epsilon'])
