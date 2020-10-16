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

import numpy as np

from extensions.front.pass_separator import FrontStart
from extensions.front.restore_ports import RestorePorts
from extensions.ops.tensor_iterator import TensorIterator
from extensions.ops.loop import Loop
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.extractor import extract_node_attrs
from mo.front.onnx.extractor import onnx_op_extractor, onnx_op_extractors
from mo.front.onnx.loader import protobuf_attrs, node_id
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, fill_graph_with_nodes, add_opoutput
from mo.ops.const import Const
from mo.ops.unsqueeze import Unsqueeze
from mo.utils.error import Error


class ONNXLoopBodyParser(FrontReplacementSubgraph):
    # The transformation updates the Loop operation node and parses body graph.
    enabled = True

    def run_before(self):
        return [FrontStart]

    def run_after(self):
        return [RestorePorts]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Loop'):
            self.create_body_graph(node)

    @staticmethod
    def create_body_graph(loop_node: Node):
        main_graph = loop_node.graph
        loop_node_name = loop_node.soft_get('name', loop_node.id)

        body_graph = Graph()
        body_graph.graph['ir_version'] = 10
        body_graph_proto = loop_node.body_proto

        initializers = Graph()
        fill_graph_with_nodes(initializers, body_graph_proto.initializer, get_id=lambda pb: pb.name, get_attrs=protobuf_attrs)

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
            elif initializers.has_node(tensor_name):
                # this is a constant
                body_graph.add_node(tensor_name, kind='op', op='Const', pb=inp, pb_init=initializers.node[tensor_name]['pb'])
            else:
                # this is a parameter
                body_graph.add_node(tensor_name, kind='op', op='Parameter', pb=inp, order=parameter_index)
                parameter_index += 1
                body_parameters.append(Node(body_graph, tensor_name))
            # add to a tensors map
            assert not tensor_name in data_nodes_map, 'Inconsistency between data_nodes_map and graph.nodes'
            data_nodes_map[tensor_name] = (tensor_name, 0)

        # go over all initializer and make sure that all of them are added to the graph
        for initializer in initializers.nodes():
            if not body_graph.has_node(initializer):
                body_graph.add_node(initializer, kind='op', op='Const', pb=initializers.node[initializer]['pb'],
                                    pb_init=initializers.node[initializer]['pb'])
                data_nodes_map[initializer] = (initializer, 0)

        # Go through all nodes in the original model order (because data nodes are defined on-the-fly and order is
        # important)
        external_edges = []
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
                    elif inp in main_graph:
                        log.debug('The edge between outer and inner graphs detected: {} -> {}'.format(inp, id))
                        # TODO FIXME the 'inp' is a tensor name of the main graph which should be propelry converted to node_name and output port
                        external_edges.append((inp, id, dst_port))  # (src node, src port),  dest node, dest port
                        continue
                    else:
                        print(inp, pb_node)
                        continue
                        raise Error(
                            'Reference to {} is not satisfied. A node refer not existing data tensor. ONNX model is not '
                            'consistent. Protobuf fragment: {}', inp, pb_node)
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
            assert body_graph.has_node(node_name), 'The body graph does not contain output with name "{}"'.format(node_name)
            body_results.append(Node(body_graph, add_opoutput(body_graph, node_name, output_port, False)))
            body_results[-1]['order'] = result_index
            result_index += 1

        # add 'internal_layer_id' attribute which is a must have attribute for TI body node
        for idx, body_node in enumerate(body_graph.get_op_nodes()):
            body_node['internal_layer_id'] = idx

#        print(external_edges)

        loop_carried_dependencies_count = len(body_graph_proto.input) - 2
        scan_outputs_count = len(body_graph_proto.output) - 1 - loop_carried_dependencies_count

        body_graph.stage = 'front'
        body_graph.graph['layout'] = 'NCHW'
        body_graph.graph['fw'] = 'onnx'
        body_graph.graph['feature_dim'] = 1
        body_graph.graph['cmd_params'] = main_graph.graph['cmd_params']

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

        # connect "trip count" input if it is not connected with default value "Infinity" (-1)
        if not loop_node.is_in_port_connected(0):
            loop_node.add_input_port(0, skip_if_exist=True)
            Const(main_graph, {'name': loop_node_name + '/trip_count', 'value': int64_array([-1])}).create_node().out_port(0).connect(loop_node.in_port(0))

        # connect "execution condition" input if it is not connected with default value True
        if not loop_node.is_in_port_connected(1):
            loop_node.add_input_port(1, skip_if_exist=True)
            Const(main_graph, {'name': loop_node_name + '/execution_condition', 'value': np.array([True], dtype=np.bool)}).create_node().out_port(0).connect(loop_node.in_port(0))

        # mark current iteration input Parameter node
        Loop.mark_current_iteration_parameter_node(loop_node, body_parameters[0])

        # connect initial value for "execution condition" input of the loop
        TensorIterator.connect_body_input(loop_node.in_port(1), body_parameters[1])
        # add back edge with "execution condition"
        TensorIterator.add_back_edge(loop_node, body_parameters[1], body_results[0])
        # mark "execution condition" Result node
        Loop.mark_execution_condition_result_node(loop_node, body_results[0])

        # connect initial value for "loop carried" dependencies variables
        for idx in range(loop_carried_dependencies_count):
            TensorIterator.connect_body_input(loop_node.in_port(idx + 2), body_parameters[idx + 2])
        # add back edge for "loop carried" dependencies variables
        for idx in range(loop_carried_dependencies_count):
            TensorIterator.add_back_edge(loop_node, body_parameters[idx + 2], body_results[idx + 1])
        # connect final value for "loop carried" dependencies variables
        for idx in range(loop_carried_dependencies_count):
            if loop_node.is_out_port_connected(idx):
                TensorIterator.connect_body_output(loop_node.out_port(idx), body_results[idx + 1])

        # connect "scan outputs" and mark axis for concatenation
        for idx in range(loop_carried_dependencies_count, loop_carried_dependencies_count + scan_outputs_count):
            unsqueeze = create_op_with_const_inputs(body_graph, Unsqueeze, {1: int64_array([0])})  # TODO add internal_layer_id???
            body_results[idx + 1].in_port(0).get_connection().insert_node(unsqueeze)
            # TODO does this approach work? Will the output values be concatenated???
            TensorIterator.connect_body_output(loop_node.out_port(idx), body_results[idx + 1], axis=0)

        # TODO add Parameters and connect nodes from outer graph
        
        # run function to parse body nodes attributes similar to the main graph
        extract_node_attrs(body_graph, lambda node: onnx_op_extractor(node, check_for_duplicates(onnx_op_extractors)))