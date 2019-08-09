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
import math
from typing import Dict

import numpy as np

from mo.front.common.partial_infer.utils import assign_dims_to_weights
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from extensions.ops.elementwise import Add, Mul

# TODO nGraph remove: rename to GEMM decomposer
class GemmToFullyConnected(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'onnx']

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('gemm', dict(kind='op', op='GEMM')),
                ('output', dict(kind='data'))],
            edges=[('gemm', 'output')]
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        log.debug('GemmToFullyConnected is triggered')
        gemm = match['gemm']
        # TODO nGraph remove BEGIN
        if not graph.graph['cmd_params'].generate_experimental_IR_V10:
            A = gemm.in_node(0)
            B = gemm.in_node(1)
            B_consumers = graph.out_edges(B.node)
            C = gemm.in_node(2)

            if not (B.value is not None and
                    C.value is not None and
                    A.shape is not None and
                    not gemm.transpose_a and
                    (len(B_consumers) == 1 or not gemm.transpose_b)):
                log.warning('Cannot convert Gemm to FullyConnected')
                return

            if gemm.transpose_b:
                # B.value = B.value.transpose()
                # B.shape = np.array(B.value.shape, dtype=np.int64)
                gemm.transpose_b = 0
            else:
                B.value = B.value.transpose()
                B.shape = np.array(B.value.shape, dtype=np.int64)

            gemm['out-size'] = gemm.out_port(0).data.get_shape()[-1]
            gemm['type'] = 'FullyConnected'
            gemm['channel_dims'] = len(match['output'].shape) - 1
            gemm['bias_addable'] = True
            gemm['input_channel_dim'] = 1  # MatMul weights in IO
            gemm['output_channel_dim'] = 0
            gemm['layout'] = 'NCHW'

            gemm.in_port(1).bin = 'weights'

        else:

            B = gemm.in_node(1)
            assert B.value is not None

            if gemm.transpose_b:
                B.value = B.value.transpose()
                B.shape = np.array(B.value.shape, dtype=np.int64)

        bias_node = Add(graph, {'name': 'MatMulBias_'}).create_node()
        gemm.out_port(0).get_connection().set_source(bias_node.out_port(0))
        gemm.in_port(2).get_connection().set_destination(bias_node.in_port(1))
        gemm.out_port(0).connect(bias_node.in_port(0))
        if graph.graph['cmd_params'].generate_experimental_IR_V10:
            gemm.type = 'MatMul'

        if gemm.has_valid('alpha'):
            if not math.isclose(gemm.alpha, 1):
                mul_node = Mul(graph, {'name': 'MatMulAlpha_'}).create_node()
                const = Const(graph, {'value': np.array(gemm.alpha)}).create_node()
                bias_node.in_port(0).get_connection().set_destination(mul_node.in_port(0))
                bias_node.in_port(0).connect(mul_node.out_port(0))
                mul_node.in_port(1).connect(const.out_port(0))
            del gemm['alpha']

        if gemm.has_valid('beta'):
            if not math.isclose(gemm.beta, 1):
                mul_node = Mul(graph, {'name': 'MatMulBeta_'}).create_node()
                const = Const(graph, {'value': np.array(gemm.beta)}).create_node()
                bias_node.in_port(1).get_connection().set_destination(mul_node.in_port(0))
                bias_node.in_port(1).connect(mul_node.out_port(0))
                mul_node.in_port(1).connect(const.out_port(0))
            del gemm['beta']

        if not graph.graph['cmd_params'].generate_experimental_IR_V10:
            assign_dims_to_weights(gemm.in_node(1), None, 1, 0, 2)
            # Do not transpose weights in this pass, it will be done as a separate pass
