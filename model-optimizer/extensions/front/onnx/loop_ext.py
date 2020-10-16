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

import logging as log

from extensions.ops.loop import Loop
from extensions.ops.parameter import Parameter
from extensions.ops.tensor_iterator import TensorIterator
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.extractor import FrontExtractorOp
from mo.front.extractor import extract_node_attrs
from mo.front.onnx.extractor import onnx_op_extractor, onnx_op_extractors
from mo.front.onnx.extractors.utils import onnx_attr
from mo.front.onnx.loader import protobuf_attrs, node_id
from mo.graph.graph import Graph, Node, fill_graph_with_nodes, add_opoutput
from mo.utils.error import Error


def connect_body_output(loop_node: Node, loop_output_port_idx: int, internal_result: Node,
                        external_node_input_ports: list = None,
                        axis: [int, None] = None, start: [int, None] = None, end: [int, None] = None,
                        stride: [int, None] = None, part_size: [int, None] = None):
    assert loop_node.soft_get('op') in ['TensorIterator', 'Loop']
    assert internal_result.soft_get('op') == 'Result'
    assert internal_result.id in loop_node.body

    loop_node.output_port_map.append({'axis': axis, 'stride': stride, 'part_size': part_size, 'start': start,
                                      'end': end, 'external_port_id': loop_output_port_idx,
                                      'internal_layer_id': internal_result['internal_layer_id']})


def connect_body_input(loop_node: Node, loop_input_port_idx: int, body_parameter: Node,
                       external_node_out_port: [int, None] = None, axis: [int, None] = None,
                       start: [int, None] = None, end: [int, None] = None, stride: [int, None] = None,
                       part_size: [int, None] = None):
    assert loop_node.soft_get('op') in ['TensorIterator', 'Loop']
    assert body_parameter.soft_get('op') == 'Parameter'
    assert body_parameter.id in loop_node.body

    loop_node.input_port_map.append({'axis': axis, 'stride': stride, 'part_size': part_size, 'start': start,
                                     'end': end, 'external_port_id': loop_input_port_idx,
                                     'internal_layer_id': body_parameter['internal_layer_id']})


class LoopExtractor(FrontExtractorOp):
    op = 'Loop'
    enabled = True

    @classmethod
    def extract(cls, loop_node):
        Loop.update_node_stat(loop_node, {'body_proto': onnx_attr(loop_node, 'body', 'g', None)})

        main_graph = loop_node.graph
        loop_node_name = loop_node.soft_get('name', loop_node.id)

        # create a Graph for body
        body_graph = Graph()
        body_graph.graph['ir_version'] = 10
        body_graph_proto = loop_node.body_proto

        initializers_graph = Graph()
        fill_graph_with_nodes(initializers_graph, body_graph_proto.initializer, get_id=lambda pb: pb.name,
                              get_attrs=protobuf_attrs)

        # maps a tensor name to a node produced it and the node port: str -> (node_id, node_port)
        data_nodes_map = {}

        # save body graph parameters and results to create the Loop connections later
        body_parameters = []
        body_results = []

        # first go through all inputs and separate constants from parameters
        parameter_index = 0
        for inp in body_graph_proto.input:
            tensor_name = str(inp.name)
            if body_graph.has_node(tensor_name):
                raise Error('Name {} of input node already exists, input names are duplicated.', tensor_name)
            elif initializers_graph.has_node(tensor_name):
                # this is a constant
                body_graph.add_node(tensor_name, kind='op', op='Const', pb=inp,
                                    pb_init=initializers_graph.node[tensor_name]['pb'])
            else:
                # this is a parameter
                body_graph.add_node(tensor_name, kind='op', op='Parameter', pb=inp, order=parameter_index)
                parameter_index += 1
                body_parameters.append(Node(body_graph, tensor_name))
            # add to a tensors map
            assert not tensor_name in data_nodes_map, 'Inconsistency between data_nodes_map and graph.nodes'
            data_nodes_map[tensor_name] = (tensor_name, 0)

        # go over all initializer and make sure that all of them are added to the graph
        for initializer in initializers_graph.nodes():
            if not body_graph.has_node(initializer):
                body_graph.add_node(initializer, kind='op', op='Const', pb=initializers_graph.node[initializer]['pb'],
                                    pb_init=initializers_graph.node[initializer]['pb'])
                data_nodes_map[initializer] = (initializer, 0)

        # Go through all nodes in the original model order because data nodes are defined on-the-fly and order matters
        external_edges = []  # (src_node, src_out_port), dest_body_parameter_node
        additional_params = {}  # (src_node, src_out_port) -> parameter_node (for newly added Parameters)
        for pb_node in body_graph_proto.node:
            # create an NX node
            id = body_graph.unique_id(node_id(pb_node))
            body_graph.add_node(id, pb=pb_node, kind='op')

            # add incoming edges based on data_nodes_map
            for dst_port, inp in enumerate(pb_node.input):
                # should add edge inp --> id
                if inp not in data_nodes_map:
                    if inp == '':
                        # input is omitted; most likely it corresponds to an optional input for an operator
                        continue
                    elif inp in main_graph.graph['tensor_mapping']:
                        log.debug('The edge between outer and inner graphs detected: {} -> {}'.format(inp, id))
                        if main_graph.graph['tensor_mapping'][inp] not in additional_params:
                            # create new Parameter in the body node and connect the body node with the outer graph using it
                            param_id = str(inp)
                            body_graph.add_node(param_id, kind='op', op='Parameter', name=param_id, pb=None, shape=None)
                            parameter_node = Node(body_graph, param_id)
                            Parameter.update_node_stat(parameter_node, {})
                            external_edges.append((main_graph.graph['tensor_mapping'][inp], parameter_node))
                            src_id, src_port = param_id, 0
                            additional_params[main_graph.graph['tensor_mapping'][inp]] = parameter_node
                        else:
                            src_id, src_port = additional_params[main_graph.graph['tensor_mapping'][inp]].id, 0
                    else:
                        raise Error('Reference to "{}" is not satisfied. A node refer not existing data tensor. ONNX '
                                    'model is not consistent. Protobuf fragment: {}', inp, pb_node)
                else:
                    src_id, src_port = data_nodes_map[inp]

                assert (body_graph.has_node(src_id))
                edge_attrs = {
                    'out': src_port,
                    'in': dst_port,
                    'name': inp,
                    'fw_tensor_debug_info': [(inp, inp)],
                    'in_attrs': ['in', 'name'],
                    'out_attrs': ['out', 'name'],
                    'data_attrs': ['fw_tensor_debug_info']
                }
                body_graph.add_edge(src_id, id, **edge_attrs)

            # add outgoing edges to data_nodes_map
            for src_port, out in enumerate(pb_node.output):
                if out in data_nodes_map:
                    log.debug("Detected reuse of blob {}.".format(out))
                data_nodes_map[out] = (id, src_port)

        result_index = 0
        for output in body_graph_proto.output:
            tensor_name = str(output.name)
            node_name, output_port = data_nodes_map[tensor_name]
            assert body_graph.has_node(node_name), 'The body graph does not contain output with name "{}"'.format(
                node_name)
            body_results.append(Node(body_graph, add_opoutput(body_graph, node_name, output_port, False)))
            body_results[-1]['order'] = result_index
            result_index += 1

        # add 'internal_layer_id' attribute which is a must have attribute for the loop body node
        for idx, body_node in enumerate(body_graph.get_op_nodes()):
            body_node['internal_layer_id'] = idx

        loop_carried_dependencies_count = len(body_graph_proto.input) - 2
        scan_outputs_count = len(body_graph_proto.output) - 1 - loop_carried_dependencies_count

        body_graph.stage = 'front'
        body_graph.graph['layout'] = 'NCHW'
        body_graph.graph['fw'] = 'onnx'
        body_graph.graph['feature_dim'] = 1
        body_graph.graph['cmd_params'] = main_graph.graph['cmd_params']
        body_graph.graph['fw_opset_version'] = main_graph.graph['fw_opset_version']

        loop_node.sub_graphs.append('body')
        loop_node['body'] = body_graph

        # Loop inputs:
        #   0 - trip count
        #   1 - execution condition
        #   2 .. - loop carried dependencies

        # Loop outputs:
        #   0 .. loop_carried_dependencies_count - 1 - loop carried dependencies
        #   loop_carried_dependencies_count .. - scan outputs

        # Body inputs:
        #   0 - iteration number
        #   1 - execution condition
        #   2 .. - loop carried dependencies

        # Body outputs:
        #   0 - execution condition
        #   1 .. loop_carried_dependencies_count - loop carried dependencies
        #   loop_carried_dependencies_count + 1 .. - scan outputs

        # TODO commented code below must be implemented in a separate transformation NormalizeLoop
        # connect "trip count" input if it is not connected with default value "Infinity" (-1)
        # if 0 not in loop_node.is_nodes(0):
        #     loop_node.add_input_port(0, skip_if_exist=True)
        #     Const(main_graph,
        #           {'name': loop_node_name + '/trip_count', 'value': int64_array([-1])}).create_node().out_port(
        #         0).connect(loop_node.in_port(0))

        # TODO commented code below must be implemented in a separate transformation NormalizeLoop
        # connect "execution condition" input if it is not connected with default value True
        # if not loop_node.is_in_port_connected(1):
        #     loop_node.add_input_port(1, skip_if_exist=True)
        #     Const(main_graph, {'name': loop_node_name + '/execution_condition',
        #                        'value': np.array([True], dtype=np.bool)}).create_node().out_port(0).connect(
        #         loop_node.in_port(0))

        # connection Loop body nodes with external input edges
        next_loop_input_port_idx = len(loop_node.in_nodes())
        for (src_node, src_port), body_node in external_edges:
            main_graph.add_edge(src_node, loop_node.id, **{'out': src_port,
                                                           'in': next_loop_input_port_idx,
                                                           'name': src_node,
                                                           'fw_tensor_debug_info': [(src_node, src_node)],
                                                           'in_attrs': ['in', 'name'],
                                                           'out_attrs': ['out', 'name'],
                                                           'data_attrs': ['fw_tensor_debug_info']}
                                )
            connect_body_input(loop_node, next_loop_input_port_idx, body_node)
            next_loop_input_port_idx += 1

        # mark current iteration input Parameter node
        Loop.mark_current_iteration_parameter_node(loop_node, body_parameters[0])

        # connect initial value for "execution condition" input of the loop
        connect_body_input(loop_node, 1, body_parameters[1])
        # add back edge with "execution condition"
        TensorIterator.add_back_edge(loop_node, body_parameters[1], body_results[0])
        # mark "execution condition" Result node
        Loop.mark_execution_condition_result_node(loop_node, body_results[0])

        # connect initial value for "loop carried" dependencies variables
        for idx in range(loop_carried_dependencies_count):
            connect_body_input(loop_node, idx + 2, body_parameters[idx + 2])
        # add back edge for "loop carried" dependencies variables
        for idx in range(loop_carried_dependencies_count):
            TensorIterator.add_back_edge(loop_node, body_parameters[idx + 2], body_results[idx + 1])
        # connect final value for "loop carried" dependencies variables
        for idx in range(loop_carried_dependencies_count):
            if idx in loop_node.out_ports():
                connect_body_output(loop_node, idx, body_results[idx + 1])

        # connect "scan outputs" and mark axis for concatenation
        for idx in range(loop_carried_dependencies_count, loop_carried_dependencies_count + scan_outputs_count):
            connect_body_output(loop_node, idx, body_results[idx + 1], axis=0)

        # run function to parse body nodes attributes similar to the main graph and create Ports
        extract_node_attrs(body_graph, lambda node: onnx_op_extractor(node, check_for_duplicates(onnx_op_extractors)))
        return cls.enabled
