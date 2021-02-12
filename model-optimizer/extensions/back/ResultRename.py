"""
 Copyright (C) 2018-2021 Intel Corporation

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
                if tensor_names:
                    result_name = tensor_names[0]
                else:
                    result_name = prev_node_out_port.node.soft_get('name', prev_node_out_port.node.id) + \
                                  '/sink_port_' + str(prev_node_out_port.idx)
                node['name'] = result_name
