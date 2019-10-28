"""
 Copyright (c) 2018-2019 Intel Corporation

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

import numpy as np

from extensions.ops.parameter import Parameter
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph, Node


class FIFOQueue(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        from extensions.front.override_batch import OverrideBatch
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
        """
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
        assert true_placeholder_shape.ndim <= 1
        if true_placeholder_shape.ndim == 1 and len(true_placeholder_shape) > 1:
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
        from extensions.front.override_batch import OverrideBatch
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

