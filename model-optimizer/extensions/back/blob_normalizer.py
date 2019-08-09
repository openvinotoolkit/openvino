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

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class BlobNormalizer(BackReplacementPattern):
    """
    This pass affects Convolution and FullyConnected weights and biases form in IR.
    Old version of those layers included weights and biases as blobs:
    <layer ... type="Convolution">
        ...
        <blobs>
            <weights offset="***" size="***"/>
            <biases offset="***" size="***"/>
        </blobs>
    </layer>

    New version (after BlobNormalizer execution) weighs and biases are represented
    as inputs to Convolution/FullyConnected layer
    """
    enabled = True

    graph_condition = [
        lambda graph: graph.graph['cmd_params'].blobs_as_inputs or
                      graph.graph['cmd_params'].generate_experimental_IR_V10
    ]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('conv', dict(type=lambda type: type in ['Convolution', 'FullyConnected']))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        conv = match['conv']
        for i in [1, 2]:
            if i in conv.in_edges() and conv.in_edges()[i] and 'bin' in conv.in_edges()[i]:
                del conv.in_edges()[i]['bin']

    def find_and_replace_pattern(self, graph: Graph):
        if graph.graph['cmd_params'].generate_experimental_IR_V10:
            for u, v, d in graph.edges(data=True):
                if 'bin' in d:
                    del d['bin']
        else:
            BackReplacementPattern.find_and_replace_pattern(self, graph)
