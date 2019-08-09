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

import copy
import logging as log

import numpy as np

from mo.front.common.layout import nhwc_to_nchw_permute
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import update_ie_fields
from mo.graph.graph import Graph
from mo.graph.graph import Node, add_opoutput
from mo.middle.replacement import MiddleReplacementPattern

nchw_to_nhwc_constant_name = 'IE_NCHW_TO_NHWC'
nhwc_to_nchw_constant_name = 'IE_NHWC_TO_NCHW'


class CustomSubgraphCall(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    @staticmethod
    def update_placeholders(graph: Graph):
        """
        Iterates over all nodes of the graph, find all TF sub-graph call operations and updates placeholders shapes and adds
        transpose operation if necessary.
        :param graph: graph to operate on
        :return: None
        """
        for node_name in graph.nodes():
            node = Node(graph, node_name)
            if node.kind == 'op' and node.has_valid('op') and node.op == 'TFCustomSubgraphCall':
                CustomSubgraphCall.update_placeholder_shape_and_add_transpose(node)

    @staticmethod
    def update_placeholder_shape_and_add_transpose(node: Node):
        """
        The function changes placeholders shapes from NHWC to NCHW format and add transpose operations if needed.
        :param node: node to operate on.
        :return: None
        """
        import tensorflow as tf
        from mo.front.common.layout import convert_shape, nhwc_to_nchw_permute, nchw_to_nhwc_permute
        from mo.front.tf.extractors.utils import tf_tensor_shape
        from mo.front.tf.partial_infer.tf import add_node_def_to_subgraph, update_input_in_pbs

        tf.reset_default_graph()

        inputs_replacements = list()

        # transpose permutation constant
        nchw_to_nhwc_constant = tf.constant(nchw_to_nhwc_permute, dtype=tf.int32, name=nchw_to_nhwc_constant_name)
        nhwc_to_nchw_constant = tf.constant(nhwc_to_nchw_permute, dtype=tf.int32, name=nhwc_to_nchw_constant_name)

        for placeholder_name in node['input_nodes_names']:
            # dummy node which we can refer to as input in the transpose for the output node
            # dummy node should be unique for each placeholder
            dummy_node = tf.constant(value=[[[[1]]]], dtype=tf.float32, name='random_dummy_name_' + placeholder_name)

            placeholder = node['pbs'][placeholder_name]
            cur_shape = tf_tensor_shape(placeholder.attr['shape'].shape)
            if len(cur_shape) == 4:  # TODO think about better check that transpose is required
                nchw_shape = convert_shape(cur_shape, nhwc_to_nchw_permute)
                for ind in range(len(cur_shape)):
                    placeholder.attr['shape'].shape.dim[ind].size = nchw_shape[ind]
                transpose_name = placeholder.name + '_transpose'
                transpose = tf.transpose(dummy_node, nchw_to_nhwc_constant, transpose_name)  # NCHW -> NHWC

                # add transpose operations to GraphDef after placeholders
                add_node_def_to_subgraph(node, transpose.op.node_def, transpose_name, len(node['input_nodes_names']))
                inputs_replacements.append((placeholder.name, transpose_name))
                inputs_replacements.append((dummy_node.name, placeholder.name))
                node['real_input_dims'].append(nchw_shape)
            else:
                node['real_input_dims'].append(cur_shape)
        add_node_def_to_subgraph(node, nchw_to_nhwc_constant.op.node_def)
        add_node_def_to_subgraph(node, nhwc_to_nchw_constant.op.node_def)

        # update initial input names to a transposed ones
        for old_input_tensor_name, new_name in inputs_replacements:
            update_input_in_pbs(node, old_input_tensor_name, new_name)

    @staticmethod
    def add_output_nodes_transposes(graph: Graph):
        """
        Iterates over all nodes of the graph, find all TF sub-graph call operations and adds Transpose operations to the
        output nodes if they are 4D to covert output from NHWC to NCHW.
        :param graph: graph to operate on
        :return: None
        """
        for node_name in graph.nodes():
            node = Node(graph, node_name)
            if node.kind == 'op' and node.has_valid('op') and node.op == 'TFCustomSubgraphCall':
                CustomSubgraphCall.add_sub_graph_call_output_tensors_transposes(node)

    @staticmethod
    def make_shape_4d(shape: np.array):
        """
        Create 4D tensor from 1D, 2D or 3D by adding new dimensions of size 1.
        :param shape: shape to extend.
        :return: 4D tensor.
        """
        new_shape = int64_array(shape)
        old_shape_len = len(shape)

        for x in range(
                4 - old_shape_len):  # TODO think about proper way to add additional dimensions considering layout
            if len(
                    new_shape) <= 1:  # if the shape is 0D or 1D then we should add additional dimensions to batch dimension
                new_shape = np.insert(new_shape, 0, 1)
            #            new_shape = np.array([1, shape[0], 1, 1])
            else:
                new_shape = np.insert(new_shape, 1, 1)
        return new_shape

    @staticmethod
    def add_reshape_before_op_node(graph: Graph, data_node_name: str, op_node_name: str, edge_attrs: dict):
        """
        Adds reshape operation which expands dimension of the specified data tensor to 4D.
        :param graph: graph to operate on.
        :param data_node_name: the name of the data node to be reshaped to 4D tensor.
        :param op_node_name: name of the TFCustomSubgraphCall node which produces the tensor.
        :param edge_attrs: edge attributes which should be preserved.
        :return: None
        """
        data_node = Node(graph, data_node_name)

        graph.remove_edge(data_node_name, op_node_name)

        assert data_node['shape'] is not None

        new_shape = CustomSubgraphCall.make_shape_4d(data_node['shape'])

        # reshape shape data node
        reshape_shape_data_node_name = graph.unique_id("Reshape_shape_")
        graph.add_node(reshape_shape_data_node_name, kind='data', precision="FP32", name=reshape_shape_data_node_name,
                       value=new_shape, shape=[1])

        # reshape operation node
        reshape_node_name = graph.unique_id("Reshape_")
        graph.add_node(reshape_node_name, kind='op', precision="FP32", type='Reshape', name=reshape_node_name,
                       op='Reshape',
                       data_type=data_node['data_type'])
        update_ie_fields(graph.node[reshape_node_name])

        # reshaped data node
        reshaped_value = None
        if data_node['value'] is not None:
            reshaped_value = np.reshape(data_node['value'], new_shape)
        reshaped_data_node_name = graph.unique_id("reshaped_data_")
        graph.add_node(reshaped_data_node_name, kind='data', precision="FP32", name=reshaped_data_node_name,
                       shape=new_shape, value=reshaped_value, nchw_layout=True)

        graph.add_edges_from([
            (data_node_name, reshape_node_name, {'in': 0}),
            (reshape_shape_data_node_name, reshape_node_name, {'in': 1}),
            (reshape_node_name, reshaped_data_node_name, {'out': 0}),
            (reshaped_data_node_name, op_node_name, edge_attrs)
        ])

    @staticmethod
    def add_reshape_after_data_node(graph: Graph, data_node_name: str):
        """
        Adds reshape operation which changes shape of the tensor produced by TFSubgraphCall from 4D to real dimension
        of the tensor. The data_node_name node contains real dimensions of the tensor but they will be changed in the
        add_reshapes_for_tf_subgraph_calls function to a 4D because IE TF call layer supports output in 4D only.
        :param graph: graph to operate on.
        :param data_node_name: name of the data node to be reshaped to correct dimensions.
        :return: None
        """
        data_node = Node(graph, data_node_name)

        # if the data node was previously marked as output then we need to mark as output new reshaped data node
        is_out_node = False
        if len(data_node.out_nodes()) == 1 and data_node.out_node().has('op') and data_node.out_node().op == 'Result':
            is_out_node = True
            graph.remove_node(data_node.out_node().id)

        # save old consumers nodes with edge attributes
        old_consumer_nodes_with_attrs = list()
        for index, out_op in enumerate(data_node.out_nodes()):
            edge_attrs = graph.get_edge_data(data_node_name, out_op.name)[0]
            old_consumer_nodes_with_attrs.append((out_op.name, edge_attrs))

        # remove old consumers from the data node
        for out_op in list(data_node.out_nodes()):
            graph.remove_edge(data_node_name, out_op.name)

        # reshape operation node
        reshape_node_name = graph.unique_id("Reshape_")
        graph.add_node(reshape_node_name, kind='op', precision="FP32", type='Reshape', name=reshape_node_name,
                       op='Reshape',
                       data_type=data_node['data_type'])
        update_ie_fields(graph.node[reshape_node_name])

        # reshape shape data node
        reshape_shape_data_node_name = graph.unique_id("Reshape_shape_")
        graph.add_node(reshape_shape_data_node_name, kind='data', precision="FP32", name=reshape_shape_data_node_name,
                       value=np.array(data_node['shape']), shape=[1])

        # reshaped data node
        reshaped_value = None
        if data_node['value'] is not None:
            reshaped_value = np.array(data_node['value'])
        reshaped_data_node_name = graph.unique_id("reshaped_data_")
        graph.add_node(reshaped_data_node_name, kind='data', precision="FP32", name=reshaped_data_node_name,
                       shape=np.array(data_node['shape']), value=reshaped_value, nchw_layout=True)

        if is_out_node:
            add_opoutput(graph, reshaped_data_node_name, 0, False)

        graph.add_edges_from([
            (data_node_name, reshape_node_name, {'in': 0}),
            (reshape_shape_data_node_name, reshape_node_name, {'in': 1}),
            (reshape_node_name, reshaped_data_node_name, {'out': 0}),
        ])

        for out_node_name, edge_attrs in old_consumer_nodes_with_attrs:
            graph.add_edges_from([
                (reshaped_data_node_name, out_node_name, edge_attrs)
            ])

    @staticmethod
    def add_reshapes_for_tf_subgraph_calls(graph: Graph):
        """
        Input and output tensors of the TFCustomSubgraphCall must be 4D because IE layer accepts and produces only 4D
        tensors. This function adds reshape operations where it is necessary.
        :param graph: graph to operate on.
        :return: None.
        """
        for src_node_name, dst_node_name, edge_attrs in list(graph.edges(data=True)):
            src_node = Node(graph, src_node_name)
            dst_node = Node(graph, dst_node_name)
            if dst_node.kind == 'op' and dst_node.has_valid('type') and dst_node.type == 'TFCustomSubgraphCall' and \
                    src_node.has_valid('shape') and len(src_node.shape) != 4:
                log.info("There is an data tensor of shape '{}' which goes into '{}' node".format(
                    src_node.shape, dst_node.type))
                CustomSubgraphCall.add_reshape_before_op_node(graph, src_node_name, dst_node_name, edge_attrs)

        for node_name in list(graph.nodes()):
            node = Node(graph, node_name)
            if node['kind'] == 'op' and node.has_and_set('type') and node.type == 'TFCustomSubgraphCall':
                for index, data_node in node.out_nodes().items():
                    real_dims_count = len(data_node.shape)
                    if real_dims_count != 4:
                        log.info(
                            "There is an data tensor of shape '{}' with real dims count '{}' which goes out of '{}' "
                            "node".format(data_node.shape, real_dims_count, node.name))
                        CustomSubgraphCall.add_reshape_after_data_node(graph, data_node.id)

                        # need to update shape of the op so IE generates XML with 4D tensors
                        out_shape = CustomSubgraphCall.make_shape_4d(data_node['shape'])

                        data_node['shape'] = out_shape

    @staticmethod
    def add_sub_graph_call_output_tensors_transposes(node: Node):
        """
        Adds transpose operations to the output nodes if they are 4D to change layout from NCHW to NHWC.
        :param node: the node to add transposes to the output nodes to.
        :return: None
        """
        import tensorflow as tf
        from mo.front.tf.partial_infer.tf import get_subgraph_output_tensors, add_node_def_to_subgraph
        _, output_tensors = get_subgraph_output_tensors(node)

        # transpose permutation constant
        nhwc_to_nchw_constant = tf.constant(nhwc_to_nchw_permute, dtype=tf.int32, name=nhwc_to_nchw_constant_name)

        # dummy node which we can refer to as input in the transpose for the output node
        dummy_node = tf.constant(value=[[[[1]]]], dtype=tf.float32, name='random_dummy_name')

        new_out_tensor_names = list()
        for out_tensor_name in node['output_tensors_names']:
            out_name, out_port = out_tensor_name.split(':')
            if len(output_tensors[
                       int(out_port)].shape) == 4:  # TODO think about better check whether transpose is required
                out_transpose_name = out_name + '_port_' + out_port + '_transpose'
                transpose = tf.transpose(dummy_node, nhwc_to_nchw_constant, name=out_transpose_name)

                # starting from TF 1.8 it is not possible to modify the "node_def" of the "tf.op", so we create a copy,
                # update it and use further
                new_input_names = transpose.op.node_def.input[:]
                new_input_names[0] = out_tensor_name
                new_node_def = copy.deepcopy(transpose.op.node_def)
                new_node_def.input[:] = new_input_names
                add_node_def_to_subgraph(node, new_node_def, position=len(node['nodes_order']))
                new_out_tensor_names.append(out_transpose_name)
            else:
                new_out_tensor_names.append(out_tensor_name)

        # update output tensor names with transposes operations
        node['output_tensors_names'] = new_out_tensor_names

    def find_and_replace_pattern(self, graph: Graph):
        CustomSubgraphCall.update_placeholders(graph)
        CustomSubgraphCall.add_output_nodes_transposes(graph)
        CustomSubgraphCall.add_reshapes_for_tf_subgraph_calls(graph)
