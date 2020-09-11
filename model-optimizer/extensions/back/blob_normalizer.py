"""
 Copyright (C) 2018-2020 Intel Corporation

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
from extensions.back.op_versioning import OpVersioning
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

    def run_before(self):
        return []

    def run_after(self):
        from extensions.back.pass_separator import BackFinish
        return [BackFinish]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('conv', dict(type=lambda type: type in ['Convolution', 'Deconvolution', 'FullyConnected']))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        conv = match['conv']
        for i in [1, 2]:
            if i in conv.in_edges() and conv.in_edges()[i] and 'bin' in conv.in_edges()[i]:
                del conv.in_edges()[i]['bin']

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes():
            if node.soft_get('type').lower() not in OpVersioning.opset_1_types and \
                    not node.soft_get('version') in ["opset2", "opset3", "opset4"]:
                continue

            for _, d in node.in_edges().items():
                if 'bin' in d:
                    del d['bin']

        for node in graph.get_data_nodes():
            for d in node.in_edges():
                if 'bin' in d:
                    del d['bin']
