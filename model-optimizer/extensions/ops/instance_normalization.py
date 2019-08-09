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

import networkx as nx

from mo.graph.graph import Graph
from mo.ops.op import Op


class InstanceNormalization(Op):
    ''' InstanceNormalization like it is defined in ONNX

        y = scale * (x - mean) / sqrt(variance + epsilon) + B

        where x is input(0), scale is input(1) and B is input(2)
    '''
    op = 'InstanceNormalization'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'epsilon': None,
            #'infer' - is not needed, this op should be replaced by a front replacer
        }, attrs)

    def supported_attrs(self):
        return ['epsilon']
