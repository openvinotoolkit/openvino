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

from extensions.ops.gather import Gather
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.reshape import Reshape


class GatherNdNormalize(MiddleReplacementPattern):
    """
    Hot fix for new speech-to-text model enabling while GatherND is not implemented in IE.
    We can replace GatherNd to Reshape + Gather in case when GatherNd indices have just one
    meaningful dimension.
    """
    enabled = True
    force_clean_up = True

    def run_before(self):
        from extensions.middle.BlockLSTMtoLSTMSequence import BlockLSTMtoLSTMSequence
        return [BlockLSTMtoLSTMSequence]

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('GatherNd', dict(kind='op', op='GatherNd'))],
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
            if not all(np.take(indices, indices=[i], axis=-1) == 0):
                if non_zero is None:
                    non_zero = i
                else:
                    return None
            else:
                if input_shape[i] != 1:
                    return None
        return non_zero

    def replace_pattern(self, graph: Graph, match: dict):
        gather = match['GatherNd']
        input_shape = gather.in_node(0).shape
        indices = gather.in_node(1).value
        if indices is None:
            # We can't do such special pass without indices value
            return

        # 0. All needed checks that we can replace GatherNd by Gather
        gather_idx = self.indices_check(indices, input_shape)
        if gather_idx is None:
            log.warning('Node {} with op=GatherNd  can\'t be normalized to op=Gather.'.format(gather.name))
            return

        # 1. Add Reshape and connect
        new_shape = int64_array([-1] + list(input_shape[indices.shape[-1]:]))
        reshape = Reshape(graph, {'name': gather.name + '/Reshape_for_GatherNd/'}).create_node()
        reshape_const_node = Const(graph, {'name': reshape.name + '/Dim', 'value': new_shape}).create_node()
        gather.in_port(0).get_connection().set_destination(reshape.in_port(0))
        reshape.in_port(1).connect(reshape_const_node.out_port(0))

        # 2. Change indices from Nd to 1d:
        new_indices = np.reshape(np.take(indices, indices=[gather_idx], axis=-1), [-1])
        new_indices_const = Const(graph, dict(value=new_indices)).create_node()

        # 3. Create new Gather operation and reconnect all inputs/outputs
        new_gather = Gather(graph, {'name': gather.name + '/NewGather/', 'axis': 0}).create_node()
        reshape.out_port(0).connect(new_gather.in_port(0))
        new_indices_const.out_port(0).connect(new_gather.in_port(1))

        gather.out_port(0).get_connection().set_source(new_gather.out_port(0))

        # 4. Remove old Gather node
        graph.remove_node(gather.id)
