# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.ops.ReduceOps import reduce_map
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.concat import Concat


class ReduceMerge(BackReplacementPattern):
    """
    Fuses sequence of Reduces of the same type into one Reduce layer of this particular type with updated axes input
    Limitations:
        - `keep_dims` attribute should be the same for all Reduces in the sequence
        - in case `keep_dims`=False: next Reduce axes should be strictly less than previous Reduce axes
    """
    enabled = True
    force_clean_up = True

    @staticmethod
    def fuse_reduces(first_reduce, second_reduce):
        first_reduce_name = first_reduce.soft_get('name', first_reduce.id)
        second_reduce_name = second_reduce.soft_get('name', second_reduce.id)
        reduce_type = first_reduce.type

        assert first_reduce.type == second_reduce.type

        if len(first_reduce.out_port(0).get_destinations()) != 1:
            # data dependency
            return

        if first_reduce.keep_dims != second_reduce.keep_dims:
            return

        first_axes = first_reduce.in_port(1).data.get_value()
        second_axes = second_reduce.in_port(1).data.get_value()
        if first_axes is None or second_axes is None:
            # dynamic axes merging is not supported
            return

        if not first_reduce.keep_dims:
            if not np.all(first_axes > second_axes):
                # indexing of upper reduce input dimensions changed
                return

        graph = second_reduce.graph

        new_axes = Concat(graph, {'name': second_reduce_name + '/Axes', 'axis': int64_array(0), 'in_ports_count': 2,
                                  'override_output_shape': True}).create_node()
        new_axes.in_port(0).connect(first_reduce.in_port(1).get_source())
        new_axes.in_port(1).connect(second_reduce.in_port(1).get_source())

        first_reduce.in_port(0).get_source().node['need_shape_inference'] = True
        first_reduce.in_port(0).get_source().node['override_output_shape'] = True

        second_reduce.in_port(1).get_connection().set_source(new_axes.out_port(0))

        first_reduce.out_port(0).get_connection().set_source(first_reduce.in_port(0).get_connection().get_source())
        first_reduce.in_port(1).disconnect()
        graph.remove_node(first_reduce.id)

        log.debug('{0} nodes {1} and {2} were fused to a single {2} node with updated axes input'
                  ''.format(reduce_type, first_reduce_name, second_reduce_name))

    def find_and_replace_pattern(self, graph: Graph):
        rsorted_nodes = graph.pseudo_topological_sort(reverse=True)
        for reduce_type in reduce_map.keys():
            reduces_of_type = [n for n in rsorted_nodes if n.id in graph and n.soft_get('type') == reduce_type]
            for second_reduce_node in reduces_of_type:
                if second_reduce_node.id not in graph:
                    continue
                first_reduce_node = second_reduce_node.in_port(0).get_source().node
                if first_reduce_node.soft_get('type', None) == reduce_type:
                    ReduceMerge.fuse_reduces(first_reduce=first_reduce_node, second_reduce=second_reduce_node)
