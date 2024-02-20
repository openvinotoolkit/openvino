# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from openvino.tools.mo.front.extractor import add_input_ops
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.convert_data_type import SUPPORTED_DATA_TYPES, np_data_type_to_precision
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern


class IteratorGetNextCut(FrontReplacementPattern):
    """
    Cuts OneShotIterator -> IteratorGetNext pattern
    in order to enable Out Of the Box (OOB) usage.
    Pass is run only if user didn't specify any inputs names and shapes.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].input is None]

    def run_before(self):
        from openvino.tools.mo.front.output_cut import OutputCut
        from openvino.tools.mo.front.input_cut import InputCut
        return [OutputCut, InputCut]

    def run_after(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        iter_get_next_shapes = defaultdict(list)
        for iter_get_next in graph.get_op_nodes(op='IteratorGetNext'):
            iter_get_next_name = iter_get_next.soft_get('name', iter_get_next.id)
            for port_idx, port in iter_get_next.out_ports().items():
                if port.disconnected():
                    continue

                if not np_data_type_to_precision(iter_get_next.types[port_idx]) in SUPPORTED_DATA_TYPES:
                    raise Error("In IteratorGetNext node '{}' data type '{}' is not supported".format(
                        iter_get_next_name, iter_get_next.types[port_idx]))

                iter_get_next_shapes[iter_get_next_name].append(dict(
                    shape=iter_get_next.shapes[port_idx],
                    out=port_idx,
                    data_type=iter_get_next.types[port_idx]
                ))

        add_input_ops(graph, iter_get_next_shapes, True)
