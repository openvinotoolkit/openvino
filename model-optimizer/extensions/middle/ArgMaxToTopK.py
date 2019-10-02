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

from extensions.ops.topk import TopK
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.const import Const


class ArgMaxToTopK(MiddleReplacementPattern):
    """
        The transformation replaces ArgMax with the TopK layer.
    """
    op = "ArgMax"
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]

    def pattern(self):
        return dict(
            nodes=[
                ('argmax', dict(op='ArgMax')),
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['argmax']

        connected_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        if len(connected_ports) == 2:
            axis = node.in_port(1).data.get_value()
        else:
            axis = node.axis

        assert axis is not None, 'The "axis" should be defined for node "{}"'.format(node.soft_get('name'))
        topk_node = TopK(graph, {'axis': axis, 'mode': 'max', 'sort': 'index'}).create_node()
        node.in_port(0).get_connection().set_destination(topk_node.in_port(0))
        if node.has_and_set('out_max_val'):  # in this mode the ArgMax produces tuples (max_ind, max_value)
            concat_node = Concat(graph, {'axis': 1, 'name': node.name + '/Concat'}).create_node()
            concat_node.add_input_port(0, skip_if_exist=True)
            concat_node.add_input_port(1, skip_if_exist=True)
            topk_node.out_port(0).connect(concat_node.in_port(1))  # indices
            topk_node.out_port(1).connect(concat_node.in_port(0))  # values
            if not node.out_port(0).disconnected():
                node.out_port(0).get_connection().set_source(concat_node.out_port(1))
        else:
            if not node.out_port(0).disconnected():
                node.out_port(0).get_connection().set_source(topk_node.out_port(1))

        topk_node.in_port(1).connect(Const(graph, {'name': node.soft_get('name') + '/TopK',
                                                   'value': node.top_k}).create_node().out_port(0))

        graph.remove_nodes_from([node.id, node.out_node(0).id])
