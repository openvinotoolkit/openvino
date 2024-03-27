# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import compatible_shapes, shape_array, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class Merge(Op):
    op = 'Merge'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'infer': self.merge_infer,
            'cf_infer': self.control_flow_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def merge_infer(node: Node):
        # we infer only through executable input nodes
        inferred_nodes = [n for n in node.in_nodes().values() if n['is_partial_inferred']]
        assert len(inferred_nodes) != 0
        tensor = inferred_nodes[0]

        if len(inferred_nodes) < len(node.in_nodes()):
            node['is_not_fully_inferred'] = True
        else:
            node['is_not_fully_inferred'] = False
            assert np.all(compatible_shapes(node.shape, inferred_nodes[0].shape) for node in inferred_nodes)

            inferred_and_executable = [n for n in node.in_nodes().values() if n['is_partial_inferred'] and
                                       'executable' in n and n['executable']]
            if len(inferred_and_executable) > 0:
                tensor = inferred_and_executable[0]

                if all([tensor.has_valid('value') and n.has_valid('value') and strict_compare_tensors(tensor.value,
                                                                                                      n.value)
                        for n in inferred_and_executable]):
                    node.out_node().value = tensor.value.copy()
                else:
                    node.out_node().value = None

        # do not use set_shape(tensor.shape) here because input port shape may be different from the calculated output
        # shape and `set_shape` will raise an error that shape has changed
        node.out_node(0).shape = shape_array(tensor.shape)

    @staticmethod
    def control_flow_infer(node: Node, is_executable: bool, mark_executability: callable):
        in_data_nodes = node.in_nodes(control_flow=True)
        out_data_nodes = node.out_nodes(control_flow=True)

        is_executable = any([d.has_and_set('executable') for i, d in in_data_nodes.items()]
                            if len(in_data_nodes) else [False])

        for i, d in out_data_nodes.items():
            mark_executability(d.id, is_executable)

