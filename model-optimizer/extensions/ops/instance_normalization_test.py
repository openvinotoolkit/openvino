# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.ops.instance_normalization import InstanceNormalization
from mo.graph.graph import Graph


class InstanceNormalizationOp(unittest.TestCase):
    def test_constructor_supported_attrs(self):
        graph = Graph()
        op = InstanceNormalization(graph, attrs={'epsilon': 0.1})
        self.assertEqual(op.supported_attrs(), ['epsilon'])
