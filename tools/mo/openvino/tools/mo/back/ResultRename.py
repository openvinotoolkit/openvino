# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class ResultRename(BackReplacementPattern):
    # This transformation sets the Result operation name equal to the incoming tensor name.
    # For some frameworks like kaldi and onnx this may result in appearance of nodes with identical names,
    # which can lead to errors in other transformations.
    # So ResultRename should be launched at the end of back phase.
    enabled = False

    def find_and_replace_pattern(self, graph: Graph):
        names_set = set()
        for node in graph.get_op_nodes():
            if node.has_valid('name'):
                names_set.add(node['name'])

        for node in graph.get_op_nodes(type='Result'):
            if node.in_ports():
                prev_node_out_port = node.in_port(0).get_connection().get_source()
                tensor_names = prev_node_out_port.get_tensor_names()
                # Graph may contain Result nodes with names equal to input tensors and
                # renaming in this case is not needed. The example of such situation is
                # IR reader check when graph is read with correct Result names.
                if not tensor_names:
                    result_name = prev_node_out_port.node.soft_get('name', prev_node_out_port.node.id) + \
                                  '/sink_port_' + str(prev_node_out_port.idx)
                    node['name'] = result_name
                    continue
                # If Result name is equal to some tensor name from list, then renaming is not needed
                if node.soft_get('name') in tensor_names:
                    continue

                # Try to find tensor name, that is not intersects with graph node names
                result_name = None
                for tensor_name in tensor_names:
                    if tensor_name not in names_set:
                        result_name = tensor_name
                        break

                # If we didn't find appropriate tensor name, then Result is named by default naming
                if result_name is None:
                    result_name = prev_node_out_port.node.soft_get('name', prev_node_out_port.node.id) + \
                                  '/sink_port_' + str(prev_node_out_port.idx)
                node['name'] = result_name
