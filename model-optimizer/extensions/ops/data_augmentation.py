# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Concat infer : N - number of inputs to concat
#                axis - dimension number for tensors concatenation
import copy

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class DataAugmentationOp(Op):
    op = 'DataAugmentation'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'extension',
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
