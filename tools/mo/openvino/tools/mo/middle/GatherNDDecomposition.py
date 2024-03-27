# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.reshape import Reshape


class GatherNDDecomposition(MiddleReplacementPattern):
    """
    Hot fix for new speech-to-text model enabling while GatherND is not implemented in IE.
    We can replace GatherND to Reshape + Gather in case when GatherND indices have just one
    meaningful dimension.
    TODO: Investigate whether we must replace GatherND with Reshape + Gather always (due to performance benefits)
          for this particular case or only if the plugin does not support GatherND.
          And the best place for the transformation is nGraph so we need to move it.
    """
    enabled = True
    force_clean_up = True

    def run_before(self):
        from openvino.tools.mo.middle.BlockLSTMtoLSTMSequence import BlockLSTMtoLSTMSequence
        return [BlockLSTMtoLSTMSequence]

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('GatherND', dict(kind='op', op='GatherND', batch_dims=0))],
            edges=[]
        )

    @staticmethod
    def indices_check(indices: np.array, input_shape: tuple):
        """
        Check that indices have just one meaningful dimension and all other dimensions of input have size 1.
        """
        n_dims = indices.shape[-1]
        non_zero = None
        for i in range(n_dims):
            if not np.all(np.take(indices, indices=[i], axis=-1) == 0):
                if non_zero is None:
                    non_zero = i
                else:
                    return None
            else:
                if input_shape[i] != 1:
                    return None
        return non_zero

    def replace_pattern(self, graph: Graph, match: dict):
        gather = match['GatherND']
        gather_name = gather.soft_get('name', gather.id)
        input_shape = gather.in_node(0).shape
        indices = gather.in_node(1).value
        if indices is None:
            # We can't do such special pass without indices value
            return

        # 0. All needed checks that we can replace GatherND by Gather
        gather_idx = self.indices_check(indices, input_shape)
        if gather_idx is None:
            log.warning(
                'Node {} with op=GatherND can\'t be normalized to op=Gather.'.format(gather_name))
            return

        # 1. Add Reshape and connect
        new_shape = int64_array([-1] + list(input_shape[indices.shape[-1]:]))
        reshape = create_op_node_with_second_input(graph, Reshape, new_shape,
                                                   {'name': gather_name + '/Reshape_for_GatherND/'})
        gather.in_port(0).get_connection().set_destination(reshape.in_port(0))

        # 2. Eliminate last dim (n_dims values) from indices shape:
        new_indices = np.reshape(
            np.take(indices, indices=[gather_idx], axis=-1), indices.shape[:-1])

        rename_node(gather, gather_name + '/to_delete')

        # 3. Create new Gather operation and reconnect all inputs/outputs
        new_gather = create_op_with_const_inputs(graph, Gather, {1: new_indices, 2: int64_array(0)},
                                                 {'name': gather_name})
        rename_node(new_gather, gather_name)

        reshape.out_port(0).connect(new_gather.in_port(0))

        gather.out_port(0).get_connection().set_source(new_gather.out_port(0))

        # 4. Remove old Gather node
        graph.remove_node(gather.id)
