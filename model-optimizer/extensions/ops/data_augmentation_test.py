# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.ops.data_augmentation import DataAugmentationOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'node_1': {'type': 'Identity', 'kind': 'op'},
    'da': {'type': 'DataAugmentation', 'kind': 'op'},
    'node_3': {'type': 'Identity', 'kind': 'op'},
    'op_output': { 'kind': 'op', 'op': 'Result'}
}


class TestConcatPartialInfer(unittest.TestCase):
    def test_tf_concat_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'da'),
                                ('da', 'node_3'),
                                ('node_3', 'op_output')
                            ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([1, 3, 227, 227])},
                                'da': {'crop_width': 225,
                                       'crop_height': 225,
                                       'write_augmented': "",
                                       'max_multiplier': 255.0,
                                       'augment_during_test': True,
                                       'recompute_mean': 0,
                                       'write_mean': "",
                                       'mean_per_pixel': False,
                                       'mean': 0,
                                       'mode': "add",
                                       'bottomwidth': 0,
                                       'bottomheight': 0,
                                       'num': 0,
                                       'chromatic_eigvec': [0.0]}
                            })

        da_node = Node(graph, 'da')
        DataAugmentationOp.data_augmentation_infer(da_node)
        exp_shape = np.array([1, 3, 225, 225])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
