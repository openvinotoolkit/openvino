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
import logging as log

from extensions.back.ResultNormalizer import ResultNormalizer
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class UselessConcatRemoval(BackReplacementPattern):
    """
    Transformation looks for the Concat nodes with just one input and remove them from the graph.
    """
    enabled = True
    run_not_recursively = True

    def run_before(self):
        return [ResultNormalizer]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('concat', {'kind': 'op', 'type': 'Concat'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        concat_node = match['concat']
        connected_ports = [port for port in concat_node.in_ports().values() if not port.disconnected()]
        if len(connected_ports) == 1:
            log.debug('Concat node {} has just one input. Removing it.'.format(concat_node.name))
            concat_node.out_port(0).get_connection().set_source(connected_ports[0].get_connection().get_source())
