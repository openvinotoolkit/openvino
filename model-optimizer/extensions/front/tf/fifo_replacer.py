"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx
import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import create_edge, erase_node
from mo.ops.input import Input


class FIFOQueue(FrontReplacementSubgraph):
    enabled = True

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[
                ('placeholder', dict(op='Placeholder', data_type=np.int32)),
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
    def replace_sub_graph(graph: nx.MultiDiGraph, match: dict, **kwargs):
        """
        Usually graph looks like:

          main_graph
            ...             OpOutput
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
        placeholder_shape = match['fifo_queue'].shape
        assert true_placeholder_shape.ndim <= 1
        if true_placeholder_shape.ndim == 1 and len(true_placeholder_shape) > 1:
            log.warning(
                'Placeholder \'{}\' got non 0-dimensional shape {} in FIFOQueue pattern. Placeholder will have the '
                'same shape after folding the pattern instead of {} shape which is original for the network.'
                ''.format(match['placeholder'].id, true_placeholder_shape, placeholder_shape))
            placeholder_shape = true_placeholder_shape
        placeholder_name = match['fifo_queue'].name
        erase_node(match['fifo_queue'])
        erase_node(match['placeholder'])
        for _, out in match['batch_join'].out_nodes().items():
            if out.id != match['image_batch'].id:
                if out.out_node().op == 'OpOutput':
                    erase_node(out.out_node())
                erase_node(out)
        erase_node(match['batch_join'])
        placeholder = Input(graph, {'name': placeholder_name, 'shape': placeholder_shape}).create_node()
        create_edge(placeholder, match['image_batch'])
        log.info("FIFOQueueV2 pattern was detected. New shape of placeholder {} is {}. Use -b to set batch size if "
                 "needed".format(placeholder.id, placeholder['shape']))
