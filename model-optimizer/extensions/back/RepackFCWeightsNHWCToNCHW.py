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

import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class RepackFCWeightsNHWCToNCHW(BackReplacementPattern):
    """
    Repack weights of FullyConnected layer as a part of nhwc_to_nchw translation if Reshape of that involves dimensions
    that we are repacking appears right before FullyConnected layer and there is a Transpose before the Reshape.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC']

    def run_after(self):
        from extensions.back.OptimizeTransposeReshapeSequence import OptimizeTransposeReshapeSequence
        return [OptimizeTransposeReshapeSequence]

    def run_before(self):
        from extensions.back.ReshapeMutation import ReshapeMutation
        from extensions.back.TransposeToPermute import TransposeToPermute
        return [ReshapeMutation, TransposeToPermute]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('permute', dict(kind='op', type='Transpose')),
                ('permute_data', dict(kind='data')),
                ('reshape', dict(kind='op', type='Reshape')),  # TODO change to reshape-like
                ('reshape_data', dict(kind='data')),
                ('fc', dict(kind='op', type='MatMul')),
            ],
            edges=[
                ('permute', 'permute_data'),
                ('permute_data', 'reshape'),
                ('reshape', 'reshape_data'),
                ('reshape_data', 'fc'),
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        fc_node = match['fc']
        permute_node = match['permute']
        reshape_node = match['reshape']
        weights = fc_node.in_node(1)

        if len(permute_node.out_port(0).get_destinations()) != 1 or \
                len(reshape_node.out_port(0).get_destinations()) != 1:
            log.debug('There are more than one consumers in the "{}" or "{}" nodes. Cannot apply transformation'.format(
                permute_node.soft_get('name'), reshape_node.soft_get('name')))
            return

        orig_shape = permute_node.in_node(0).shape
        new_shape = reshape_node.out_node().shape

        # OK, here we are; need to repack fc_node.in_node(1) to maintain it compatible with original input order

        assert all(orig_shape != -1), 'Input shape for {} can not be negative.'.format(fc_node.id)
        assert all(new_shape != -1), 'Output shape for {} can not be negative.'.format(fc_node.id)
        assert fc_node.in_node(1).has_valid('value'), 'Node {} does not have value.'.format(fc_node.id)

        log.debug("orig_shape = {}".format(orig_shape))
        log.debug("new_shape = {}".format(new_shape))
        log.debug("weights.shape = {}".format(weights.shape))
        log.debug("weights.shape[1] = {}, new_shape[1] = {}".format(weights.shape[1], new_shape[1]))

        if len(orig_shape) != 4 or len(new_shape) != 2 or orig_shape[0] != new_shape[0] or \
                np.prod(orig_shape[1:]) != new_shape[1]:
            log.debug('Cannot merge Transpose node "{}" into the FullyConnected node "{}" because of origin and new '
                      'shapes mismatch'.format(permute_node.soft_get('name'), fc_node.soft_get('name')))
            return

        if weights.shape[0] != new_shape[1]:
            log.debug('First dim of weights does not correspond to output shape of {}'.format(fc_node.id))
            return

        # interpret I dimension of the weights as packed HWC
        order = permute_node.in_port(1).data.get_value()
        tmp_shape = np.concatenate([orig_shape[order[1:]], [weights.shape[1]]])
        weights.value = np.transpose(weights.value.reshape(tmp_shape), (2, 0, 1, 3)).reshape(weights.shape)

        # remove Transpose node and data node
        permute_node.in_port(0).get_connection().set_destination(reshape_node.in_port(0))
        graph.remove_nodes_from([permute_node.id, permute_node.out_node(0).id])
