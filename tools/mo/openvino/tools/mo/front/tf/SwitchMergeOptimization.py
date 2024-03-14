# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.select import Select
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph


class SwitchMergeOptimization(FrontReplacementSubgraph):
    """
    Optimization for case, when combination of Switches have one common condition and can be expressed as Select node.

    This transformation matches too big number of instances for models with many BatchNorm layers with the same input
    from the model input data node with training/inference flag. So the transformation is implemented as a simple graph
    traversal instead of regular pattern-based approach.
    
    The following pattern is checked:
        nodes=[('Merge', dict(kind='op', op='Merge')),
               ('Switch_2_input', dict(kind='data')),
               ('Switch_2', dict(kind='op', op='Switch')),
               ('Switch_2_data', dict(kind='data')),
               ('op', dict(kind='op')),
               ('op_data', dict(kind='data')),
               ('Switch', dict(kind='op', op='Switch')),
               ('Switch_data', dict(kind='data')),
               ('Switch_1', dict(kind='op', op='Switch')),
               ('Switch_1_data', dict(kind='data')),
               ('cond_data', dict(kind='data')),
               ('identity', dict(kind='op', op='Identity')),
               ('identity_data', dict(kind='data')),
               ],
        edges=[
               ('Switch_2_input', 'Switch_2', {'in': 0}),
               ('Switch_2', 'Switch_2_data', {'out': 1}),
               ('Switch_2_data', 'Merge'),
               ('cond_data', 'Switch_2', {'in': 1}),
               ('cond_data', 'Switch_1', {'in': 1}),
               ('cond_data', 'Switch', {'in': 1}),
               ('Switch_1', 'Switch_1_data', {'out': 0}),
               ('Switch', 'Switch_data', {'out': 0}),
               ('Switch_1_data', 'op', {'in': 1}),
               ('Switch_data', 'op', {'in': 0}),
               ('op', 'op_data'),
               ('op_data', 'identity'),
               ('identity', 'identity_data'),
               ('identity_data', 'Merge'),
               ],
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for merge in graph.get_op_nodes(op='Merge'):
            for merge_switch_in_port in range(2):
                if merge.in_port(merge_switch_in_port).disconnected() or \
                        merge.in_port(merge_switch_in_port).get_source().node.op != 'Switch':
                    continue
                switch_2 = merge.in_port(merge_switch_in_port).get_source().node

                if merge.in_port(1 - merge_switch_in_port).disconnected() or \
                        merge.in_port(1 - merge_switch_in_port).get_source().node.op != 'Identity':
                    continue
                false_value_port = merge.in_port(1 - merge_switch_in_port).get_source()

                true_value_port = switch_2.in_port(0).get_source()
                op = false_value_port.node.in_port(0).get_source().node

                if op.in_port(0).disconnected() or op.in_port(0).get_source().node.op != 'Switch':
                    continue
                switch = op.in_port(0).get_source().node

                if op.in_port(1).disconnected() or op.in_port(1).get_source().node.op != 'Switch':
                    continue
                switch_1 = op.in_port(1).get_source().node

                if switch.in_port(1).get_source() == switch_1.in_port(1).get_source() and \
                        switch.in_port(1).get_source() == switch_2.in_port(1).get_source():
                    select = Select(graph, dict(name=merge.soft_get('name') + '/Select/', format='tf')).create_node()
                    select.in_port(0).connect(switch.in_port(1).get_source())
                    select.in_port(1).connect(true_value_port)
                    select.in_port(2).connect(false_value_port)

                    merge.out_port(0).get_connection().set_source(select.out_port(0))

                    assert 1 in op.in_ports() and 0 in op.in_ports()

                    op.in_port(0).disconnect()
                    op.in_port(1).disconnect()

                    switch.in_port(0).get_connection().set_destination(op.in_port(0))
                    switch_1.in_port(0).get_connection().set_destination(op.in_port(1))

                    graph.remove_nodes_from(nodes=[switch_1.id, switch.id, switch_2.id, merge.id])
                    # need to exit from the inner for loop because the Merge op has been removed
                    break
