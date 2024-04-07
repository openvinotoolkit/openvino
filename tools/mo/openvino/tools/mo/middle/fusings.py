# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.div import Div
from openvino.tools.mo.front.sub import Sub
from openvino.tools.mo.middle.AddFakeQuantizeFuse import AddFakeQuantizeFuse
from openvino.tools.mo.middle.EltwiseInputReshape import normalize_eltwise_inputs
from openvino.tools.mo.middle.MulFakeQuantizeFuse import MulFakeQuantizeFuse
from openvino.tools.mo.middle.RemoveRedundantReshapes import RemoveRedundantReshapes

from openvino.tools.mo.middle.pass_separator import PostMiddleStart
from openvino.tools.mo.middle.quantize_fuses import MarkNodesToFuseUpToFakeQuantize, FakeQuantizeFuse
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.conv import fuse_pad
from openvino.tools.mo.middle.passes.fusing.decomposition import convert_scale_shift_to_mul_add, convert_batch_norm
from openvino.tools.mo.middle.passes.fusing.fuse_grouped_conv import grouped_convolutions_fusing
from openvino.tools.mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from openvino.tools.mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from openvino.tools.mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes, mark_shape_of_sugraph_as_unfusable
from openvino.tools.mo.middle.passes.fusing.resnet_optimization import stride_optimization
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class Fusing(MiddleReplacementPattern):
    enabled = True
    replacement_id = "fusing"
    force_clean_up = True
    run_not_recursively = True

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def run_before(self):
        # the Fusing transformation adds Reshape layers in some cases which could be removed by the
        # RemoveRedundantReshapes transformation
        return [PostMiddleStart, RemoveRedundantReshapes]

    def find_and_replace_pattern(self, graph: Graph):
        fw = graph.graph['fw']
        argv = graph.graph['cmd_params']
        layout = graph.graph['layout']

        mark_shape_of_sugraph_as_unfusable(graph)
        for_graph_and_each_sub_graph_recursively(graph, fuse_pad)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        # Converting FusedBatchNorm layer to Mul->Add->Mul->Add sequence
        # IE doesn't support batchNormInference with 4 inputs, so we have to split it to two ScaleShift
        for_graph_and_each_sub_graph_recursively(graph, convert_batch_norm)

        if fw == 'caffe':
            # Converting ScaleShift layer to Mul->Add
            for_graph_and_each_sub_graph_recursively(graph, convert_scale_shift_to_mul_add)

        for_graph_and_each_sub_graph_recursively(graph, Div().find_and_replace_pattern)
        for_graph_and_each_sub_graph_recursively(graph, Sub().find_and_replace_pattern)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        if fw != 'caffe':
            # Converting ScaleShift layer to Mul->Add
            for_graph_and_each_sub_graph_recursively(graph, convert_scale_shift_to_mul_add)
            for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        # Fusing the sequences of Mul/Add operations
        for_graph_and_each_sub_graph_recursively(graph, fuse_mul_add_sequence)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        normalize_eltwise_inputs(graph)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        # Fusing linear operation to Convolution
        for_graph_and_each_sub_graph_recursively(graph, fuse_linear_ops)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        for_graph_and_each_sub_graph_recursively(graph, grouped_convolutions_fusing)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        for_graph_and_each_sub_graph_recursively(graph, fuse_linear_ops)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        for_graph_and_each_sub_graph_recursively(graph, normalize_eltwise_inputs)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        MarkNodesToFuseUpToFakeQuantize().find_and_replace_pattern(graph)
        FakeQuantizeFuse().find_and_replace_pattern(graph)
        AddFakeQuantizeFuse().find_and_replace_pattern(graph)
        MulFakeQuantizeFuse().find_and_replace_pattern(graph)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        mark_shape_of_sugraph_as_unfusable(graph)
        for_graph_and_each_sub_graph_recursively(graph, fuse_pad)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        if layout != 'NHWC':
            stride_optimization(graph)
