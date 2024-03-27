# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Graph


class PackBinaryWeights(BackReplacementPattern):
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', type='BinaryConvolution'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        conv = match['op']
        assert len(conv.in_nodes()) == 2
        initial_shape = conv.in_port(1).data.get_shape()
        assert initial_shape is not None
        weights = conv.in_port(1).data.get_value().flatten()
        weights_rounded = np.round(weights)
        assert np.all(np.isclose(weights, weights_rounded))
        assert len(conv.in_node(1).out_nodes()) == 1
        weights_rounded = mo_array(weights_rounded, dtype=np.int32) + 1  # -1 --> 0
        # Reversing element in chunks by 8 elements to pack bits correctly
        # First need to pad data with necessary number of element to make the length dividable by 8
        pad = (-len(weights_rounded)) % 8
        weights_rounded = mo_array(np.concatenate((weights_rounded, np.zeros([pad]))), dtype=np.int32)
        assert len(weights_rounded) % 8 == 0
        weights_rounded = weights_rounded.reshape([len(weights_rounded) // 8, 8])
        weights_rounded = np.flip(weights_rounded, axis=1)
        weights_rounded = weights_rounded.flatten()
        packed = np.packbits(weights_rounded)
        conv.in_port(1).data.set_value(packed)
        conv['packed_weights'] = 1

        conv.in_node(1)['force_shape'] = initial_shape.copy()
        conv.in_node(1)['shape'] = initial_shape.copy()
        conv.in_node(1)['force_type'] = 'U1'
