# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class ResultRename(BackReplacementPattern):
    # This transformation sets the Result operation name equal to the incoming tensor name.
    # For some frameworks like kaldi and onnx this may result in appearance of nodes with identical names,
    # which can lead to errors in other transformations.
    # So ResultRename should be launched at the end of back phase.
    enabled = False

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='Result'):
            if node.in_ports():
                prev_node_out_port = node.in_port(0).get_connection().get_source()
                tensor_names = prev_node_out_port.get_tensor_names()
                # Graph may contain Result nodes with names equal to input tensors and
                # renaming in this case is not needed. The example of such situation is
                # IR reader check when graph is read with correct Result names.
                if tensor_names and node.soft_get('name') == tensor_names[0]:
                    continue
                if tensor_names and not graph.get_op_nodes(name=tensor_names[0]):
                    result_name = tensor_names[0]
                else:
                    result_name = prev_node_out_port.node.soft_get('name', prev_node_out_port.node.id) + \
                                  '/sink_port_' + str(prev_node_out_port.idx)
                node['name'] = result_name
