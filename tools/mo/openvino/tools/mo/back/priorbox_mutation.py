# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.back.ForceStrictPrecision import ForceStrictPrecision
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.strided_slice import StridedSlice
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class PriorboxMutation(BackReplacementPattern):
    enabled = True
    force_shape_inference = True

    def run_before(self):
        return [ForceStrictPrecision]

    def pattern(self):
        return dict(
            nodes=[
                ('pb', {'type': lambda node_type: node_type in ['PriorBox', 'PriorBoxClustered']})
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['pb']
        name = node.soft_get('name', node.id)

        graph.graph['cmd_params'].static_shape = False

        assert len(node.in_ports()) == 2

        begin = Const(graph, {'value': mo_array([2], dtype=np.int32), 'name': name + '/ss_begin'}).create_node()
        end = Const(graph, {'value': mo_array([4], dtype=np.int32), 'name': name + '/ss_end'}).create_node()
        stride = Const(graph, {'value': mo_array([1], dtype=np.int32), 'name': name + '/ss_stride'}).create_node()

        shape_0 = Shape(graph, {'name': name + '/0_port'}).create_node()
        ss_0 = StridedSlice(graph, {'name': name + '/ss_0_port',
                                    'begin_mask': mo_array([1], dtype=np.int32),
                                    'end_mask': mo_array([0], dtype=np.int32),
                                    'new_axis_mask': mo_array([0], dtype=np.int32),
                                    'shrink_axis_mask': mo_array([0], dtype=np.int32),
                                    'ellipsis_mask': mo_array([0], dtype=np.int32)}).create_node()

        shape_0.out_port(0).connect(ss_0.in_port(0))
        begin.out_port(0).connect(ss_0.in_port(1))
        end.out_port(0).connect(ss_0.in_port(2))
        stride.out_port(0).connect(ss_0.in_port(3))

        source = node.in_port(0).get_connection().get_source()
        node.in_port(0).disconnect()
        source.connect(shape_0.in_port(0))
        ss_0.out_port(0).connect(node.in_port(0))

        shape_1 = Shape(graph, {'name': name + '/1_port'}).create_node()
        ss_1 = StridedSlice(graph, {'name': name + '/ss_1_port',
                                    'begin_mask': mo_array([1], dtype=np.int32),
                                    'end_mask': mo_array([0], dtype=np.int32),
                                    'new_axis_mask': mo_array([0], dtype=np.int32),
                                    'shrink_axis_mask': mo_array([0], dtype=np.int32),
                                    'ellipsis_mask': mo_array([0], dtype=np.int32)}).create_node()

        shape_1.out_port(0).connect(ss_1.in_port(0))
        begin.out_port(0).connect(ss_1.in_port(1))
        end.out_port(0).connect(ss_1.in_port(2))
        stride.out_port(0).connect(ss_1.in_port(3))

        source = node.in_port(1).get_connection().get_source()
        node.in_port(1).disconnect()
        source.connect(shape_1.in_port(0))
        ss_1.out_port(0).connect(node.in_port(1))

        ss_0['force_precision_in_ports'] = {1: 'int64', 2: 'int64', 3: 'int64'}
        ss_1['force_precision_in_ports'] = {1: 'int64', 2: 'int64', 3: 'int64'}

        node['need_shape_inference'] = True
        node['override_output_shape'] = True
        node['V10_infer'] = True
        unsqueeze = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]), {'name': name + '/unsqueeze'})
        naked_priorbox_name = name + '/naked_not_unsqueezed'
        rename_nodes([(node, naked_priorbox_name), (unsqueeze, name)])

        node.out_port(0).get_connection().set_source(unsqueeze.out_port(0))
        node.out_port(0).connect(unsqueeze.in_port(0))
