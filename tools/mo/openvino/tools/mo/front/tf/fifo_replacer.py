# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from collections import defaultdict

import numpy as np

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph, FrontReplacementPattern
from openvino.tools.mo.front.extractor import add_input_ops
from openvino.tools.mo.front.output_cut import OutputCut
from openvino.tools.mo.front.user_data_repack import UserDataRepack
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_precision, SUPPORTED_DATA_TYPES
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.utils.error import Error


class FIFOQueue(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.override_batch import OverrideBatch
        return [OverrideBatch]

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[
                ('placeholder', dict(op='Parameter', data_type=np.int32)),
                ('fifo_queue', dict(op='FIFOQueueV2')),
                ('batch_join', dict(op='QueueDequeueUpToV2')),
                ('image_batch', dict(op='Identity', data_type=np.float32))
            ],
            edges=[
                ('placeholder', 'batch_join', {'out': 0}),
                ('fifo_queue', 'batch_join', {'out': 0}),
                ('batch_join', 'image_batch', {'out': 0})
            ]
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict, **kwargs):
        r"""
        Usually graph looks like:

          main_graph
            ...             Result
             |                 |
        image_batch      label_batch
                \        /
                batch_join
                /        \
        placeholder      fifo_queue

        Replacer works for both cases (that's why we have loop - 68 line):
            label_batch was marked as output
            there is no label_batch node
        """
        true_placeholder_shape = match['placeholder'].shape
        placeholder_shape = match['fifo_queue'].shapes[0]
        placeholder_data_type = match['fifo_queue'].types[0]
        # in case OOB conversion batch_size placeholder shape is not required
        # so use a shape specified in FIFOQueueV2 shapes list attribute
        assert true_placeholder_shape is None or true_placeholder_shape.ndim <= 1
        if true_placeholder_shape is not None and true_placeholder_shape.ndim == 1 and len(true_placeholder_shape) > 1:
            log.warning(
                'Placeholder \'{}\' got non 0-dimensional shape {} in FIFOQueue pattern. Placeholder will have the '
                'same shape after folding the pattern instead of {} shape which is original for the network.'
                ''.format(match['placeholder'].id, true_placeholder_shape, placeholder_shape))
            placeholder_shape = true_placeholder_shape
        placeholder_name = match['fifo_queue'].name
        graph.erase_node(match['fifo_queue'])
        graph.erase_node(match['placeholder'])
        for _, out in match['batch_join'].out_nodes().items():
            if out.id != match['image_batch'].id:
                if out.out_node().op == 'Result':
                    graph.remove_node(out.out_node().id)
                graph.remove_node(out.id)
        graph.remove_node(match['batch_join'].id)
        placeholder = Parameter(graph, {'name': placeholder_name, 'shape': placeholder_shape,
                                        'data_type': placeholder_data_type}).create_node()
        graph.create_edge(placeholder, match['image_batch'])
        log.info("FIFOQueueV2 pattern was detected. New shape of placeholder {} is {}. Use -b to set batch size if "
                 "needed".format(placeholder.id, placeholder['shape']))


class QueueDequeueManyV2(FrontReplacementSubgraph):
    """
    Replaces the combination of the FIFOQueueV2 + QueueDequeueManyV2 operations with a number of Placeholders.
    """
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.override_batch import OverrideBatch
        return [OverrideBatch]

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[
                ('fifo_queue', dict(op='FIFOQueueV2')),
                ('queue_deque', dict(op='QueueDequeueManyV2')),
            ],
            edges=[
                ('fifo_queue', 'queue_deque', {'out': 0}),
            ]
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict, **kwargs):
        inputs_dict = {}
        for u, v, edge_attrs in graph.out_edges(match['queue_deque'].id, data=True):
            out_port = edge_attrs['out']
            shape = match['fifo_queue'].shapes[out_port]
            if out_port not in inputs_dict:
                input_op = Parameter(graph, {'shape': shape.copy()})
                inputs_dict[out_port] = input_op.create_node([])
            graph.create_edge(inputs_dict[out_port], Node(graph, v), edge_attrs['out'], edge_attrs['in'], edge_attrs)

        graph.remove_node(match['queue_deque'].id)
        graph.remove_node(match['fifo_queue'].id)


class FIFOQueueDequeueCut(FrontReplacementPattern):
    """
    Cuts FIFOQueue -> QueueDequeue pattern in order to enable Out Of the Box (OOB) usage.
    Pass runs only if user didn't specify any input names and shapes.
    This transformation relies on output shapes and types extracted from QueueDequeue node.
    In the meantime, the transformations FIFOQueue and QueueDequeueManyV2 expects output shapes and types extracted
    from FIFOQueue node.
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

            new_inputs = ""
            fifo_qd_name = node.soft_get('name', node.id)
            for port_idx, port in node.out_ports().items():
                if port.disconnected():
                    continue
                if not np_data_type_to_precision(node.types[port_idx]) in SUPPORTED_DATA_TYPES:
                    raise Error("Data type {} is not supported for the"
                                "node {}".format(node.types[port_idx], fifo_qd_name))

                fifo_qd_shapes[fifo_qd_name].append(dict(
                    shape=node.shapes[port_idx],
                    out=port_idx,
                    data_type=node.types[port_idx]
                ))
                new_inputs += "{}:{}, ".format(fifo_qd_name, port_idx)

            log.error(
                "Found TF {} operation in the model. "
                "PLEASE NOTE, the model will contain new input(s) ".format(node.op)
                + new_inputs +
                "created due to automatically triggered pruning transformation for this operation.",
                extra={'is_warning': True}
            )

        add_input_ops(graph, fifo_qd_shapes, True)
