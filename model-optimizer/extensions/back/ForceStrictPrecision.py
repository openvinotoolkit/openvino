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
from extensions.ops.Cast import Cast
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import data_type_str_to_np


class ForceStrictPrecision(BackReplacementPattern):
    """
    Assign precision for some inputs for specific layers depending on their semantics.

    To identify ports which should be processed, this pass relies on special attributes
    inside a node: force_precision_in_ports. This attribute should be a dictionary with
    index of port as key and required precision code as value (e.g. 'int64' etc.).
    """
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('node', {'force_precision_in_ports': lambda x: x is not None})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['node']
        for in_port, precision in node.force_precision_in_ports.items():
            if in_port in node.in_ports().keys() and not node.in_port(in_port).disconnected():
                cast = Cast(graph, {'name': node.name + '/Cast_' + str(in_port),
                                    'dst_type': data_type_str_to_np(precision)}).create_node()
                node.in_port(in_port).get_connection().insert_node(cast)
