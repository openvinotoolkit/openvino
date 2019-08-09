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

import numpy as np

from extensions.back.ReshapeMutation import ReshapeMutation
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.reshape import Reshape


class MatMulNormalizer(BackReplacementPattern):
    """ Work-around for incorrect models (likely from ONNX) where GEMM takes tensors with rank > 2

        ONNX/Gemm supports 2D inputs only. We observe some models that are constructed in a way where
        GEMM has an input with 4D input. It should be flattenned before.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_before(self):
        return [ReshapeMutation]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', type='MatMul'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        input_shape = node.in_port(0).data.get_shape()
        if len(input_shape) > 2:
            new_shape = Const(graph, {'value': np.array([0, -1], dtype=np.int64)}).create_node()
            reshape = Reshape(graph, {}).create_node()
            source = node.in_port(0).get_source()
            node.in_port(0).get_connection().set_source(reshape.out_port(0))
            source.connect(reshape.in_port(0))
            new_shape.out_port(0).connect(reshape.in_port(1))
            new_shape.infer(new_shape)
            reshape.infer(reshape)
