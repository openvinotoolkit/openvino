# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class ResultRename(BackReplacementPattern):
    # This transformation sets the Result operation name equal to the incoming tensor name.
    # For some frameworks like kaldi and onnx this may result in appearance of nodes with identical names,
    # which can lead to errors in other transformations.
    # So ResultRename should be launched at the end of back phase.
    enabled = False

    def find_and_replace_pattern(self, graph: Graph):
        op_names = set()
        result_names_map = dict()
        for node in graph.get_op_nodes():
            if node.has_valid('name'):
                op_names.add(node['name'])

        for node in graph.get_op_nodes(type='Result'):
            if node.in_ports():
                prev_node_out_port = node.in_port(0).get_connection().get_source()
                tensor_names = prev_node_out_port.get_tensor_names()
                # Graph may contain Result nodes with names equal to input tensors and
                # renaming in this case is not needed. The example of such situation is
                # IR reader check when graph is read with correct Result names.
                if node.soft_get('name') in tensor_names:
                    continue

                # Try to find tensor name, that is not intersects with graph node names
                result_name = None
                for tensor_name in tensor_names:
                    if tensor_name not in op_names:
                        if node.has_valid('name'):
                            op_names.remove(node['name'])
                        op_names.add(tensor_name)
                        result_name = tensor_name
                        break

                # If we didn't find appropriate tensor name, then Result is named by default naming
                if result_name is None:
                    result_name = prev_node_out_port.node.soft_get('name', prev_node_out_port.node.id) + \
                                  '/sink_port_' + str(prev_node_out_port.idx)
                    log.warning("Tensor name for Result node with name {} wasn't found. "
                                "Default renaming was used: {}".format(node.soft_get('name', node.id),
                                                                       result_name))
                result_names_map[node['name']] = result_name
                node['name'] = result_name

        # Change names saved in graph.outputs_order
        for i in range(len(graph.outputs_order)):
            if graph.outputs_order[i] in result_names_map:
                graph.outputs_order[i] = result_names_map[graph.outputs_order[i]]
