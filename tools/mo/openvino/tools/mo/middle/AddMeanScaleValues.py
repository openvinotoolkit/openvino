# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.layout import get_dim_from_layout, get_features_dim
from openvino.tools.mo.front.common.partial_infer.utils import compatible_dims
from openvino.tools.mo.front.extractor import get_node_id_with_ports
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.elementwise import Add, Mul
from openvino.tools.mo.utils.cli_parser import get_node_name_with_port_from_input_value
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class AddMeanScaleValues(MiddleReplacementPattern):
    enabled = True
    run_not_recursively = True

    def run_after(self):
        return []

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    @staticmethod
    def insert_pre_processing(graph: Graph, input_node: Node, node_mean_scale_values: np.array,
                              preprocessing_name: str):
        assert preprocessing_name in ['scale', 'mean']
        if node_mean_scale_values.get(preprocessing_name) is None:
            return
        user_value = node_mean_scale_values[preprocessing_name]
        value = 1 / user_value if preprocessing_name == 'scale' else user_value * (-1)
        optimize_value = int(preprocessing_name == 'scale')
        op = Mul if preprocessing_name == 'scale' else Add

        if all([x == optimize_value for x in value]):
            return
        assert input_node.has_valid('shape')
        in_name = input_node.soft_get('name', input_node.id)
        features_dim_idx, has_layout = get_dim_from_layout(input_node, 'C')
        if features_dim_idx is None:
            if has_layout:
                log.warning('Layout for input {} doesn\'t have channel ("C") dimension to apply {} preprocessing. '
                            'Skipping this input.'.format(in_name, preprocessing_name))
            features_dim_idx = get_features_dim(graph.graph['layout'], len(input_node.shape))
        assert compatible_dims(value.size, input_node.shape[features_dim_idx]) or value.size == 1, \
            "Incompatible layout, please specify correct layout for the node"

        shape = np.ones(len(input_node.shape), dtype=np.int64)
        shape[features_dim_idx] = value.size
        value = value.reshape(shape)

        if input_node.op == 'Parameter' and input_node.has_and_set('data_type'):
            dtype = input_node.data_type
            if np.issubdtype(dtype, np.floating):
                value = value.astype(dtype)

        name = in_name + '/' + preprocessing_name
        preprocessing = create_op_with_const_inputs(graph, op=op, port_value_dict={1: value}, op_attrs={'name': name})

        if input_node.is_out_port_connected(0) and len(input_node.out_port(0).get_destinations()) == 1:
            # There are models with pattern Parameter(uint8) -> Convert(float).
            # Adding mean/scale leads to the following:
            # Parameter(uint8) -> Mean/Scale -> Convert(float) which is incorrect.
            # To fix this mean and scale preprocessing node is inserted after Convert(float) node.
            out_node = input_node.out_port(0).get_destination().node
            convert_type = out_node.soft_get('dst_type')
            if out_node.soft_get('type') == "Convert" and (convert_type in [np.float32, np.float16]):
                input_node = out_node
                if convert_type != value.dtype:
                    new_value = value.astype(convert_type)
                    const_node = preprocessing.in_port(1).get_connection().get_source().node
                    const_node['value'] = new_value

        for dst in input_node.out_port(0).get_destinations():
            if dst.node.soft_get('type') != 'ShapeOf':
                # After the insertion of additional operations model optimizer
                # should keep the link to the input layer. Parameter node in framework
                # should map to parameter node in IR.
                # For this reason 'fw_tensor_debug_info' should be kept in data node.
                dst.get_connection().set_source(preprocessing.out_port(0), "source")

        input_node.out_port(0).connect(preprocessing.in_port(0))

    @staticmethod
    def apply_scale(graph: Graph, input_node: Node, node_mean_scale_values: dict):
        AddMeanScaleValues.insert_pre_processing(graph, input_node, node_mean_scale_values, preprocessing_name='scale')

    @staticmethod
    def apply_mean_value(graph: Graph, input_node: Node, node_mean_scale_values: dict):
        AddMeanScaleValues.insert_pre_processing(graph, input_node, node_mean_scale_values, preprocessing_name='mean')

    def find_and_replace_pattern(self, graph: Graph):
        values = graph.graph['cmd_params'].mean_scale_values
        input_nodes = graph.get_op_nodes(op='Parameter')

        if not isinstance(values, dict):
            # The case when input names to apply mean/scales weren't specified
            if len(values) != len(input_nodes):
                raise Error('Numbers of inputs and mean/scale values do not match. ' + refer_to_faq_msg(61))

            data = np.copy(values)
            values = {}
            for idx, node in enumerate(input_nodes):
                values.update(
                    {
                        node.soft_get('name', node.id): {
                            'mean': data[idx][0],
                            'scale': data[idx][1]
                        }
                    }
                )

        for node_name, node_mean_scale_values in values.items():
            node_id = None
            node_name = get_node_name_with_port_from_input_value(node_name)
            try:
                node_id, direction, port = get_node_id_with_ports(graph, node_name, skip_if_no_port=False)
                assert direction != 'out', 'Only input port can be specified for mean/scale application'
            except Error as e:
                log.warning('node_name {} is not found in graph'.format(node_name))
            if Node(graph, node_id) not in input_nodes:
                # if the user cutted-off input of the network then input node name specified in the --scale_values
                # or --mean_values doesn't correspond to a real input node generated by Model Optimizer. But
                # the information about initial input node name is stored in Placeholder's attribute 'initial_node_name'
                new_node_id = None
                for placeholder in input_nodes:
                    try:
                        placeholder_port = int(placeholder.id.split("_")[-1])
                    except Exception as ex:
                        log.debug('Can not get the port number from the node {}'.format(placeholder.id))
                        log.debug('Port will be defined as None')
                        port = None
                    if placeholder.has('initial_node_name') and placeholder.initial_node_name == node_id and (
                            port is None or placeholder_port == port):
                        new_node_id = placeholder.id
                        break
                if new_node_id is None:
                    raise Error('Input with name {} wasn\'t found!'.format(node_name) +
                                refer_to_faq_msg(83))
                node_id = new_node_id

            input_node = Node(graph, node_id)
            AddMeanScaleValues.apply_scale(graph, input_node, node_mean_scale_values)
            AddMeanScaleValues.apply_mean_value(graph, input_node, node_mean_scale_values)
