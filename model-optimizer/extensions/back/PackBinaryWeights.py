"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log

import networkx as nx
import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Node, Graph
from mo.ops.tile import Tile


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
        weights = conv.in_port(1).data.get_value().flatten()
        weights_rounded = np.round(weights)
        assert np.all(np.isclose(weights, weights_rounded))
        assert len(conv.in_node(1).out_nodes()) == 1
        weights_rounded = np.array(weights_rounded, dtype=np.int32) + 1  # -1 --> 0
        # Reversing element in chunks by 8 elements to pack bits correctly
        # First need to pad data with necessary number of element to make the length dividable by 8
        pad = (-len(weights_rounded))%8
        weights_rounded = np.array(np.concatenate((weights_rounded, np.zeros([pad]))), dtype=np.int32)
        assert len(weights_rounded) % 8 == 0
        weights_rounded = weights_rounded.reshape([len(weights_rounded)//8, 8])
        weights_rounded = np.flip(weights_rounded, axis=1)
        weights_rounded = weights_rounded.flatten()
        packed = np.packbits(weights_rounded)
        conv.in_port(1).data.set_value(packed)
        conv.in_node(1)['force_precision'] = 'uint8'
        conv['packed_weights'] = 1