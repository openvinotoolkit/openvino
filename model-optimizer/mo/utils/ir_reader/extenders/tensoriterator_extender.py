"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender
from mo.utils.ir_reader.layer_to_class import copy_graph_with_ops


class TensorIterator_extender(Extender):
    op = 'TensorIterator'

    @staticmethod
    def extend(op: Node):

        def normalize_port_map(port_map: dict):
            for port in port_map:
                for elem in ['axis', 'stride', 'part_size', 'start', 'end']:
                    if port.get(elem) is None:
                        port[elem] = None

        assert op.has('body'), 'Something wrong with TensorIterator layer {}, please check!'.format(op.name)

        # Now op.body is an IREngine, we need to replace it with IREngine.graph
        op.body.graph.graph['cmd_params'] = op.graph.graph['cmd_params']
        op.body.graph.graph['ir_version'] = op.graph.graph['ir_version']
        op.body.graph.name = op.name + '/body'

        for node in op.body.graph.get_op_nodes():
            node['internal_layer_id'] = int(node.id)

        op.body = copy_graph_with_ops(op.body.graph)

        normalize_port_map(op.input_port_map)
        normalize_port_map(op.output_port_map)

        for edge in op.back_edges:
            edge['from_layer'] = edge['from-layer']
            edge['to_layer'] = edge['to-layer']

            del(edge['from-layer'])
            del(edge['to-layer'])

        op['infer'] = Extender.const_shape_infer
