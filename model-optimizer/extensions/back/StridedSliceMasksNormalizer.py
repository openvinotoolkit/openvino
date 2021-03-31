# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph


class StridedSliceMasksNormalizer(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.back.ConvolutionNormalizer import DeconvolutionNormalizer
        from extensions.back.CropToStridedSlice import CropToStridedSlice
        return [CropToStridedSlice, DeconvolutionNormalizer]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='StridedSlice'):
            assert node.has_valid('begin_mask')
            assert node.has_valid('end_mask')
            node.begin_mask = int64_array([1 - i for i in node.begin_mask])
            node.end_mask = int64_array([1 - i for i in node.end_mask])
