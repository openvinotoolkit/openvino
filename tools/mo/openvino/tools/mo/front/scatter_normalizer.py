# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


class ScatterNormalizer(FrontReplacementPattern):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(is_scatter=True):
            name = node.soft_get('name', node.id)
            input_ports_count = len([port for port in node.in_ports().values() if not port.disconnected()])
            has_axis = node.has_valid('axis')

            if has_axis:
                assert input_ports_count == 3, \
                    '{} node {} has unexpected number of input ports {}'.format(node.op, name, input_ports_count)
                const = Const(graph, {'name': name + '/axis', 'value': np.int64(node.axis)}).create_node()
                node.add_input_port(3, skip_if_exist=True)
                node.in_port(3).connect(const.out_port(0))
                del node['axis']
            else:
                assert input_ports_count == 4, \
                    '{} node {} has unexpected number of input ports {}'.format(node.op, name, input_ports_count)
