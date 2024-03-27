# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.rank_decomposer import RankDecomposer
from openvino.tools.mo.ops.ReduceOps import ReduceMax, ReduceMean
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.ops.rank import Rank
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


class GlobalPoolingToReduce(FrontReplacementPattern):
    op = "Pooling"
    enabled = True

    pool_method_to_reduce_type = {
        'max': ReduceMax,
        'avg': ReduceMean,
    }

    def run_before(self):
        return [RankDecomposer]

    def find_and_replace_pattern(self, graph: Graph):
        global_poolings = graph.get_op_nodes(type='Pooling', global_pool=True)
        if len(global_poolings) == 0:
            return

        layout = graph.graph['layout']
        assert layout != 'NHWC', 'Global pooling transformation depends on layout (NHWC not enabled)'

        for pooling in global_poolings:
            name = pooling.soft_get('name', pooling.id)
            assert pooling.has_valid('pool_method'), 'Global Pooling {} has no `pool_method` attribute'.format(name)
            method = pooling['pool_method']
            assert method in self.pool_method_to_reduce_type, \
                'Unexpected Global Pooling method `{}` for node `{}`'.format(method, name)
            reduce_op_class = self.pool_method_to_reduce_type[method]

            reduce = reduce_op_class(graph, {'name': name + '/reduce', 'keep_dims': True}).create_node()

            pooling.out_port(0).get_connection().set_source(reduce.out_port(0))
            src = pooling.in_port(0).get_connection().get_source()

            reduce.in_port(0).get_connection().set_source(src)

            start = Const(graph, {'value': int64_array(2)}).create_node()
            end = Rank(graph, {'name': name + '/input_rank'}).create_node()
            delta = Const(graph, {'value': int64_array(1)}).create_node()

            axis = Range(graph, {'name': name + '/global_pooling_reduce_axis'}).create_node()

            axis.in_port(0).connect(start.out_port(0))
            src.connect(end.in_port(0))
            axis.in_port(1).connect(end.out_port(0))
            axis.in_port(2).connect(delta.out_port(0))

            axis.out_port(0).connect(reduce.in_port(1))

            log.debug('Global {} pooling was converted to reduce: `{}`'.format(method, name))
