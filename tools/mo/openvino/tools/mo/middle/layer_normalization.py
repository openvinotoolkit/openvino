# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.elementwise import Mul, Add
from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class LayerNormalization(MiddleReplacementPattern):
    """
    Decompose LayerNorm(x) to MVN(x) * gamma + beta
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='LayerNorm'):
            node_name = node.soft_get('name', node.id)
            input_shape = node.in_port(0).data.get_shape()
            assert node.has_valid('axis'), 'Incorrect axis value for the node {}'.format(node_name)
            axis = node.axis

            mvn = create_op_node_with_second_input(graph, MVN, int64_array([axis]),
                                                   dict(eps=node.epsilon, name=node_name + '/LayerNorm/MVN_',
                                                        across_channels=1, normalize_variance=1, eps_mode='inside_sqrt'))

            mul = Mul(graph, {'name': node_name + '/LayerNorm/mul_'}).create_node()
            add = Add(graph, {'name': mul.name + '/LayerNorm/add_'}).create_node()

            node.in_port(0).get_connection().set_destination(mvn.in_port(0))
            node.in_port(1).get_connection().set_destination(mul.in_port(1))
            node.in_port(2).get_connection().set_destination(add.in_port(1))

            mvn.out_port(0).connect(mul.in_port(0))
            mul.out_port(0).connect(add.in_port(0))
            node.out_port(0).get_connection().set_source(add.out_port(0))

            # MXNet LayerNorm gamma and beta attributes are 1D tensors with shape = [input_shape[axis]]
            # We have to unsqueeze values for Mul and Add operations to avoid shapes incompatibility problems
            # if axis != -1
            canonical_axis = get_canonical_axis_index(input_shape, axis)
            unsqueeze_value = []
            for idx, val in enumerate(input_shape):
                if idx != canonical_axis:
                    unsqueeze_value.append(idx)

            mul_const_unsqueeze = create_op_node_with_second_input(graph, Unsqueeze,
                                                                   int64_array(unsqueeze_value),
                                                                   dict(name=mul.name + '/Unsqueeze',
                                                                        override_output_shape=True))
            add_const_unsqueeze = create_op_node_with_second_input(graph, Unsqueeze,
                                                                   int64_array(unsqueeze_value),
                                                                   dict(name=add.name + '/Unsqueeze',
                                                                        override_output_shape=True))

            mul.in_port(1).get_connection().insert_node(mul_const_unsqueeze)
            add.in_port(1).get_connection().insert_node(add_const_unsqueeze)

            rename_nodes([(node, node_name + '/ShouldBeDeleted'), (add, node_name)])
