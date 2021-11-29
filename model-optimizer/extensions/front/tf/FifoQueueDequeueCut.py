# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict

from extensions.front.output_cut import OutputCut
from extensions.front.user_data_repack import UserDataRepack
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.extractor import add_input_ops
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import np_data_type_to_precision, SUPPORTED_DATA_TYPES
from mo.utils.error import Error


class FifoQueueDequeueCut(FrontReplacementPattern):
    """
    Cuts FifoQueue -> QueueDequeue pattern
    in order to enable Out Of the Box (OOB) usage.
    Pass is run only if user didn't specify any inputs names and shapes.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].input is None]

    def run_before(self):
        return [OutputCut]

    def run_after(self):
        return [UserDataRepack]

    def find_and_replace_pattern(self, graph: Graph):
        fifo_qd_shapes = defaultdict(list)
        for node in graph.get_op_nodes():
            if node.op not in ["QueueDequeue", "QueueDequeueV2"]:
                continue

            fifo_qd_name = node.soft_get('name', node.id)
            for port in node.out_nodes().keys():
                if not np_data_type_to_precision(node.types[port]) in SUPPORTED_DATA_TYPES:
                    raise Error("Data type {} is not supported for the node {}".format(node.types[port], fifo_qd_name))

                fifo_qd_shapes[fifo_qd_name].append(dict(
                    shape=node.shapes[port],
                    out=port,
                    data_type=node.types[port]
                ))

        add_input_ops(graph, fifo_qd_shapes, True)
