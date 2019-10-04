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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging as log

from extensions.back.CreateConstNodes import CreateConstNodesReplacement
from extensions.back.FuseReshapesSequence import FuseReshapesSequence
from extensions.back.RemoveRedundantReshapes import RemoveRedundantReshapes
from extensions.middle.AddFakeQuantizeFuse import AddFakeQuantizeFuse
from extensions.middle.EltwiseInputNormalization import EltwiseInputNormalize
from extensions.middle.MulFakeQuantizeFuse import MulFakeQuantizeFuse
from extensions.middle.quantize_fuses import MarkNodesToFuseUpToFakeQuantize, FakeQuantizeFuse
from mo.front.common.register_custom_ops import update_extractors_with_extensions, check_for_duplicates
from mo.front.extractor import extract_node_attrs, remove_output_ops
from mo.front.onnx.extractor import onnx_op_extractor, onnx_op_extractors
from mo.front.onnx.loader import load_onnx_model, protobuf2nx
from mo.middle.passes.conv import convert_add_or_mul_to_scaleshift, convert_muladd_to_scaleshift, fuse_pad, \
    convert_matmul_to_fully_connected
from mo.middle.passes.eliminate import graph_clean_up_onnx, remove_const_ops
from mo.middle.passes.fusing.decomposition import convert_batch_norm, convert_scale_shift_to_mul_add
from mo.middle.passes.fusing.fuse_grouped_conv import grouped_convolutions_fusing
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.passes.shape import reverse_input_channels, merge_nodes_permutations, permute_data_nodes_attrs, \
    permute_op_nodes_attrs
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.pipeline.common import prepare_emit_ir
from mo.utils import class_registration
from mo.utils.cli_parser import get_meta_info
from mo.utils.error import Error
from mo.utils.logger import log_step
from mo.utils.utils import refer_to_faq_msg


def driver(argv: argparse.Namespace, model_file_name: str, output_model_name: str, output_dir: str):
    log_step(argv.steps, 'LOAD')
    meta_info = get_meta_info(argv)

    model_proto = load_onnx_model(model_file_name)
    model_graph = model_proto.graph  # pylint: disable=no-member
    # print(model_graph)
    # assert len(model_graph) == 1, "An ONNX model contains more than 1 graph: unsupported"
    log.debug("Number of nodes in graph_def: {}".format(len(model_graph.node)))
    log.debug("Number of all input ports (not true inputs) in graph_def: {}".format(len(model_graph.input)))
    log.debug("Number of initializers in graph_def: {}".format(len(model_graph.initializer)))
    log.debug("Number of real inputs in graph_def: {}".format(len(model_graph.input) - len(model_graph.initializer)))
    update_extractors_with_extensions(onnx_op_extractors)

    try:
        graph = protobuf2nx(model_proto)
        log.debug("Number of nodes in NX graph: {}".format(graph.number_of_nodes()))
        graph.__setattr__('name',
                          output_model_name if output_model_name else model_proto.graph.name)  # pylint: disable=no-member
        graph.graph['layout'] = 'NCHW'
        graph.graph['cmd_params'] = argv
        graph.graph['fw'] = 'onnx'
        graph.graph['feature_dim'] = 1 if graph.graph['layout'] == 'NCHW' else 3

        if graph.graph['cmd_params'].generate_experimental_IR_V10:
            version = 10
        else:
            version = 6
        graph.graph['ir_version'] = 2 if argv.generate_deprecated_IR_V2 else version

    except Exception as e:
        raise Error(
            'Cannot pre-process ONNX graph after reading from model file "{}". ' \
            'File is corrupt or has unsupported format. Details: {}. ' +
            refer_to_faq_msg(44),
            model_file_name,
            str(e)
        ) from e
    graph.check_empty_graph('protobuf2nx. It may happen due to problems with loaded model')
    extract_node_attrs(graph, lambda node: onnx_op_extractor(node, check_for_duplicates(onnx_op_extractors)))

    # --------------------------------- LOAD END ------------------------------------------------------
    log_step(argv.steps, 'FRONT')
    class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)
    log_step(argv.steps, 'MIDDLE')
    class_registration.apply_replacements(graph, class_registration.ClassType.MIDDLE_REPLACER)

    fuse_pad(graph)
    graph_clean_up_onnx(graph)

    for_graph_and_each_sub_graph_recursively(graph, convert_matmul_to_fully_connected)

    # Mark nodes with attr 'can_be_fused': False to disable fusing for specified nodes
    mark_unfused_nodes(graph, argv.finegrain_fusing)

    # Converting FusedBatchNorm layer to Mul->Add->Mul->Add sequence
    # IE doesn't support BN with 4 inputs, so we have to split it to two ScaleShift
    convert_batch_norm(graph)
    graph_clean_up_onnx(graph)

    if not argv.disable_fusing:
        # Converting ScaleShift layer to Mul->Add
        convert_scale_shift_to_mul_add(graph)
        graph_clean_up_onnx(graph)

        # Fusing the sequences of Mul/Add operations
        fuse_mul_add_sequence(graph)
        graph_clean_up_onnx(graph)

        # Fusing linear operation to Convolution
        fuse_linear_ops(graph)
        graph_clean_up_onnx(graph)

    if not argv.disable_gfusing:
        grouped_convolutions_fusing(graph)
        graph_clean_up_onnx(graph)
        if not argv.disable_fusing:
            fuse_linear_ops(graph)
            graph_clean_up_onnx(graph)

    MarkNodesToFuseUpToFakeQuantize().find_and_replace_pattern(graph)
    FakeQuantizeFuse().find_and_replace_pattern(graph)

    AddFakeQuantizeFuse().find_and_replace_pattern(graph)
    MulFakeQuantizeFuse().find_and_replace_pattern(graph)

    convert_muladd_to_scaleshift(graph)
    graph_clean_up_onnx(graph)

    graph_clean_up_onnx(graph)
    convert_add_or_mul_to_scaleshift(graph)  # scale = 1
    graph_clean_up_onnx(graph)

    fuse_pad(graph)
    graph_clean_up_onnx(graph)

    if argv.reverse_input_channels:
        reverse_input_channels(graph)

    if argv.move_to_preprocess:
        move_scaleshift_to_preprocess(graph)
        graph_clean_up_onnx(graph)

    FuseReshapesSequence().find_and_replace_pattern(graph)
    RemoveRedundantReshapes().find_and_replace_pattern(graph)

    graph_clean_up_onnx(graph)

    pattern = EltwiseInputNormalize()
    pattern.find_and_replace_pattern(graph)

    merge_nodes_permutations(graph)
    permute_data_nodes_attrs(graph)
    permute_op_nodes_attrs(graph)

    graph_clean_up_onnx(graph)

    log_step(argv.steps, 'BACK')
    class_registration.apply_replacements(graph, class_registration.ClassType.BACK_REPLACER)

    for_graph_and_each_sub_graph_recursively(graph, remove_const_ops)

    CreateConstNodesReplacement().find_and_replace_pattern(graph)

    for_graph_and_each_sub_graph_recursively(graph, remove_output_ops)

    log_step(argv.steps, 'EMIT')
    prepare_emit_ir(graph=graph, data_type=argv.data_type, output_dir=output_dir, output_model_name=output_model_name,
                    meta_info=meta_info)

    return 0
