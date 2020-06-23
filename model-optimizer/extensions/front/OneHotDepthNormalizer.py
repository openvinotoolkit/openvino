"""
 Copyright (C) 2020 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.ops.reshape import Reshape


class OneHotDepthNormalizer(FrontReplacementPattern):
    """
    Transformation performs squeezeng one-element tensors on 1st input in OneHot into 0D scalars. This transformation
    allows to avoid problems with some models produced by tf2onnx which have 1D depth in OneHot.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('onehot', dict(kind='op', type='OneHot'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['onehot']
        node_name = node.soft_get('name', node.id)
        reshape = create_op_with_const_inputs(graph, Reshape, {1: int64_array([])}, {'name': node_name + '/Reshape'})
        node.in_port(1).get_connection().insert_node(reshape)
