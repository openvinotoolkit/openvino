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

from extensions.back.TopKNormalizer import TopKNormalizer
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class ResultRename(BackReplacementPattern):
    enabled = True

    def run_after(self):
        return [TopKNormalizer]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('result', {'type': 'Result'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['result']
        if node.in_ports():
            prev_node_out_port = node.in_port(0).get_connection().get_source()
            result_name = prev_node_out_port.get_tensor_names(first_only=True)
            if result_name is None:
                result_name = prev_node_out_port.node.name + '/sink_port_' + str(prev_node_out_port.idx)
            node['name'] = result_name
