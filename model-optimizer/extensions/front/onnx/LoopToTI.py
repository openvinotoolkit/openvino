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

from extensions.front.pass_separator import FrontStart
from extensions.front.restore_ports import RestorePorts
from extensions.ops.range import Range
from extensions.ops.tensor_iterator import TensorIterator
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.extractor import extract_node_attrs
from mo.front.onnx.extractor import onnx_op_extractor, onnx_op_extractors
from mo.front.onnx.loader import protobuf_attrs, node_id
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, fill_graph_with_nodes, add_opoutput
from mo.ops.unsqueeze import Unsqueeze
from mo.utils.error import Error


class LoopToTI(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [FrontStart]

    def run_after(self):
        return [RestorePorts]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Loop'):
            self.convert_to_ti(node)

    @staticmethod
    def convert_to_ti(node: Node):
        main_graph = node.graph
        body_graph = Graph()
        graph_pb = node.body_proto

        initializers = Graph()
        fill_graph_with_nodes(initializers, graph_pb.initializer, get_id=lambda pb: pb.name, get_attrs=protobuf_attrs)

        # maps a tensor name to a node produced it and the node port: str -> (node_id, node_port)
        data_nodes_map = {}

        # save body graph parameters and results to create the TI connections later
        internal_parameters = []
        internal_results = []

        # first go through all inputs and separate constant from placeholders
        for inp in graph_pb.input:
            tensor_name = str(inp.name)
            if body_graph.has_node(tensor_name):
                raise Error('Name {} of input node already exists, input names are duplicated.', tensor_name)
            elif initializers.has_node(tensor_name):
                # this is a constant
                body_graph.add_node(tensor_name, kind='op', op='Const', pb=inp, pb_init=initializers.node[tensor_name]['pb'])
            else:
                # this is a placeholder
                body_graph.add_node(tensor_name, kind='op', op='Parameter', pb=inp)
                internal_parameters.append(Node(body_graph, tensor_name))
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
        for pb_node in graph_pb.node:
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

        for output in graph_pb.output:
            tensor_name = str(output.name)
            node_name, output_port = data_nodes_map[tensor_name]
            assert body_graph.has_node(node_name), 'The body graph does not contain output with name "{}"'.format(node_name)
            internal_results.append(Node(body_graph, add_opoutput(body_graph, node_name, output_port, False)))

        # add 'internal_layer_id' attribute which is a must have attribute for TI body node
        for idx, body_node in enumerate(body_graph.get_op_nodes()):
            body_node['internal_layer_id'] = idx

        print(external_edges)
        log.debug('')

        loop_carried_dependencies_count = len(graph_pb.input) - 2
        scan_outputs_count = len(graph_pb.output) - 1 - loop_carried_dependencies_count

        body_graph.stage = 'front'
        # create TI node and connect inputs
        loop_node_name = node.soft_get('name', node.id)
        ti = TensorIterator(main_graph, {'name': loop_node_name + '/TensorIterator', 'body': body_graph}).create_node()
        # TI inputs:
        # 0 - iteration number
        # 1 - loop condition
        # 2 .. - loop carried dependencies
        for idx in range(2 + loop_carried_dependencies_count):
            ti.add_input_port(idx)

        if node.is_in_port_connected(0):
            iterations = create_op_with_const_inputs(main_graph, Range, {0: int64_array(0), 2: int64_array(1)},
                                                     {'name': loop_node_name + '/IterationsRange'})
            TensorIterator.add_input(ti.in_port(0), internal_parameters[0], iterations.out_port(0), 0, -1, 1, 1)

        if node.is_in_port_connected(1):
            TensorIterator.add_input(ti.in_port(1), internal_parameters[1], node.in_port(1).get_source())
            TensorIterator.add_back_edge(ti, internal_parameters[1], internal_results[0])

        for idx in range(loop_carried_dependencies_count):
            TensorIterator.add_input(ti.in_port(idx + 2), internal_parameters[idx + 1], node.in_port(idx + 2).get_source())

        # TI outputs:
        # 0 - condition
        # 1 .. loop_carried_dependencies_count + 1 - loop carried dependencies
        # loop_carried_dependencies_count + 2 .. - scan outputs
        for idx in range(loop_carried_dependencies_count + scan_outputs_count):
            ti.add_output_port(idx)

        for idx in range(loop_carried_dependencies_count):
            if node.is_out_port_connected(idx):
                TensorIterator.add_output(ti.out_port(idx), internal_results[idx + 1],
                                          node.out_port(idx).get_destinations())

        for idx in range(loop_carried_dependencies_count, loop_carried_dependencies_count + scan_outputs_count):
            unsqueeze = create_op_with_const_inputs(body_graph, Unsqueeze, {1: int64_array(0)})
            internal_results[idx + 1].in_port(0).get_connection().insert_node(unsqueeze)
            # TODO does this approach work? Will the output values be concatenated???
            if node.is_out_port_connected(idx):
                TensorIterator.add_output(ti.out_port(idx),
                                          internal_results[idx + 1], node.out_port(idx).get_destinations(), axis=0)



        extract_node_attrs(body_graph, lambda node: onnx_op_extractor(node, check_for_duplicates(onnx_op_extractors)))

        # now when we replaced the Loop node with the TI node we can safely remove Loop node
        main_graph.remove_node(node.id)
        # main_graph.dump_graph_for_graphviz(save_to_svg=True)
        # exit(0)