# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.back.ResultNormalizer import ResultNormalizer
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


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
