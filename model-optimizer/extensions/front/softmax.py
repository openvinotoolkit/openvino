"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.front.div import Div
from extensions.front.reduce_axis_normalizer import ReduceAxisNormalizer
from extensions.front.sub import Sub
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph
from mo.ops.softmax import Softmax


class SoftmaxFromKeras(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern that Keras produces for SoftMax layer. The transformation works if the
    softmax is performed over one pre-defined axis.
    """
    enabled = True

    def run_before(self):
        return [Sub, Div]

    def run_after(self):
        return [ReduceAxisNormalizer]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('reduce_max', dict(op='ReduceMax')),
                ('reduce_indices_max', dict(op='Const', value=lambda x: x is not None and x.size != 0)),
                ('sub', dict(op='Sub')),
                ('exp', dict(op='Exp')),
                ('reduce_sum', dict(op='ReduceSum')),
                ('reduce_indices_sum', dict(op='Const', value=lambda x: x is not None and x.size != 0)),
                ('div', dict(op='Div')),
            ],
            edges=[
                ('input', 'sub', {'in': 0}),
                ('input', 'reduce_max', {'in': 0}),
                ('reduce_indices_max', 'reduce_max', {'in': 1}),
                ('reduce_max', 'sub', {'in': 1}),
                ('sub', 'exp', {'in': 0}),
                ('exp', 'div', {'in': 0}),
                ('exp', 'reduce_sum', {'in': 0}),
                ('reduce_indices_sum', 'reduce_sum', {'in': 1}),
                ('reduce_sum', 'div', {'in': 1}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):

        reduce_max_axis = match['reduce_indices_max'].value
        reduce_sum_axis = match['reduce_indices_sum'].value

        if reduce_max_axis.ndim == 0:
            reduce_max_axis = reduce_max_axis.reshape([1])

        if reduce_sum_axis.ndim == 0:
            reduce_sum_axis = reduce_sum_axis.reshape([1])

        if len(reduce_max_axis) != 1:
            log.info('The reductions indices contain more than 1 element. Cannot convert to Softmax.')
            return

        if not np.array_equal(reduce_max_axis, reduce_sum_axis):
            log.info('The reduce indices are not equal: {} vs {}. Cannot convert to Softmax'
                     ''.format(reduce_max_axis, reduce_sum_axis))
            return

        softmax = Softmax(graph, {'name': match['input'].name + '/Softmax', 'axis': reduce_sum_axis[0]}).create_node()
        match['input'].out_port(0).connect(softmax.in_port(0))
        match['div'].out_port(0).get_connection().set_source(softmax.out_port(0))

        log.debug('Successfully created SoftMax node')
