# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.op_versioning import OpVersioning
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


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
        from openvino.tools.mo.back.pass_separator import BackFinish
        return [BackFinish]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('conv', dict(type=lambda type: type in ['Convolution', 'Deconvolution', 'FullyConnected', 'DeformableConvolution']))],
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
                    not node.soft_get('version') in ["opset2", "opset3", "opset4", "opset8"]:
                continue

            for _, d in node.in_edges().items():
                if 'bin' in d:
                    del d['bin']

        for node in graph.get_data_nodes():
            for d in node.in_edges():
                if 'bin' in d:
                    del d['bin']
