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
from extensions.front.div import Div
from extensions.front.sub import Sub
from extensions.middle.AddFakeQuantizeFuse import AddFakeQuantizeFuse
from extensions.middle.EltwiseInputNormalization import EltwiseInputNormalize
from extensions.middle.MulFakeQuantizeFuse import MulFakeQuantizeFuse
from extensions.middle.RemoveRedundantReshapes import RemoveRedundantReshapes

from extensions.middle.pass_separator import PostMiddleStart
from extensions.middle.quantize_fuses import MarkNodesToFuseUpToFakeQuantize, FakeQuantizeFuse
from mo.graph.graph import Graph
from mo.middle.passes.conv import fuse_pad
from mo.middle.passes.fusing.decomposition import convert_scale_shift_to_mul_add, convert_batch_norm
from mo.middle.passes.fusing.fuse_grouped_conv import grouped_convolutions_fusing
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.fusing.resnet_optimization import stride_optimization
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.middle.replacement import MiddleReplacementPattern


class Fusing(MiddleReplacementPattern):
    enabled = True
    replacement_id = "fusing"
    force_clean_up = True
    run_not_recursively = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def run_before(self):
        # the Fusing transformation adds Reshape layers in some cases which could be removed by the
        # RemoveRedundantReshapes transformation
        return [PostMiddleStart, RemoveRedundantReshapes]

    def find_and_replace_pattern(self, graph: Graph):
        fw = graph.graph['fw']
        argv = graph.graph['cmd_params']
        layout = graph.graph['layout']

        for_graph_and_each_sub_graph_recursively(graph, fuse_pad)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        # Mark nodes with attr 'can_be_fused': False to disable fusing for specified nodes
        for_graph_and_each_sub_graph_recursively(graph, lambda graph: mark_unfused_nodes(graph, argv.finegrain_fusing))

        # Converting FusedBatchNorm layer to Mul->Add->Mul->Add sequence
        # IE doesn't support BN with 4 inputs, so we have to split it to two ScaleShift
        for_graph_and_each_sub_graph_recursively(graph, convert_batch_norm)

        if fw == 'caffe':
            # Converting ScaleShift layer to Mul->Add
            for_graph_and_each_sub_graph_recursively(graph, convert_scale_shift_to_mul_add)

        for_graph_and_each_sub_graph_recursively(graph, Div().find_and_replace_pattern)
        for_graph_and_each_sub_graph_recursively(graph, Sub().find_and_replace_pattern)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        if not argv.disable_fusing:
            if fw != 'caffe':
                # Converting ScaleShift layer to Mul->Add
                for_graph_and_each_sub_graph_recursively(graph, convert_scale_shift_to_mul_add)
                for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

            # Fusing the sequences of Mul/Add operations
            for_graph_and_each_sub_graph_recursively(graph, fuse_mul_add_sequence)
            for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

            EltwiseInputNormalize().find_and_replace_pattern(graph)
            for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

            # Fusing linear operation to Convolution
            for_graph_and_each_sub_graph_recursively(graph, fuse_linear_ops)
            for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        if not argv.disable_gfusing:
            for_graph_and_each_sub_graph_recursively(graph, grouped_convolutions_fusing)
            for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())
            if not argv.disable_fusing:
                for_graph_and_each_sub_graph_recursively(graph, fuse_linear_ops)
                for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        EltwiseInputNormalize().find_and_replace_pattern(graph)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        MarkNodesToFuseUpToFakeQuantize().find_and_replace_pattern(graph)
        FakeQuantizeFuse().find_and_replace_pattern(graph)
        AddFakeQuantizeFuse().find_and_replace_pattern(graph)
        MulFakeQuantizeFuse().find_and_replace_pattern(graph)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        for_graph_and_each_sub_graph_recursively(graph, fuse_pad)
        for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        if layout != 'NHWC' and not argv.disable_resnet_optimization:
            stride_optimization(graph)
