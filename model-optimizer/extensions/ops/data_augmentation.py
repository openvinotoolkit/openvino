"""
 Copyright (c) 2017-2019 Intel Corporation

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

# Concat infer : N - number of inputs to concat
#                axis - dimension number for tensors concatenation
import copy

import networkx as nx

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class DataAugmentationOp(Op):
    op = 'DataAugmentation'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': DataAugmentationOp.data_augmentation_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'crop_width',
            'crop_height',
            'write_augmented',
            'max_multiplier',
            'augment_during_test',
            'recompute_mean',
            'write_mean',
            'mean_per_pixel',
            'mean',
            'mode',
            'bottomwidth',
            'bottomheight',
            'num',
            'chromatic_eigvec'
        ]

    @staticmethod
    def data_augmentation_infer(node: Node):
        outn = node.out_node(0)
        inn = node.in_node(0)

        outn.shape = copy.copy(inn.shape)

        if node.crop_width != 0 or node.crop_height != 0:
            outn.shape[2] = node.crop_height
            outn.shape[3] = node.crop_width
