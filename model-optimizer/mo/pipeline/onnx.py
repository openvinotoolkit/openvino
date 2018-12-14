"""
 Copyright (c) 2018 Intel Corporation

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

import numpy as np

from extensions.middle.EltwiseInputNormalization import EltwiseInputNormalize
from extensions.middle.NormalizeFullyConnected import NormalizeFullyConnected
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.extractor import add_output_ops, add_input_ops, \
    extract_node_attrs, create_tensor_nodes, remove_output_ops, user_data_repack
from mo.front.onnx.extractor import common_onnx_fields, onnx_op_extractor, onnx_op_extractors
from mo.front.onnx.loader import load_onnx_model, protobuf2nx
from mo.middle.passes.conv import convert_add_to_scaleshift, convert_gemm_to_fully_connected, \
    convert_muladd_to_scaleshift_or_power, fuse_pad, convert_dilated_convolution, convert_mul_to_scaleshift
from mo.middle.passes.eliminate import graph_clean_up, remove_op_nodes, remove_useless_split
from mo.middle.passes.fusing.decomposition import convert_batch_norm, convert_scale_shift_to_mul_add
from mo.middle.passes.fusing.fuse_grouped_conv import grouped_convolutions_fusing
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.infer import scale_input, override_placeholder_shapes, partial_infer, convert_mul_add_to_power, \
    update_fully_connected_shapes, add_mean_scale_values, override_batch
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.passes.shape import convert_reshape, reverse_input_channels, \
    fuse_sequence_of_reshapes, merge_nodes_permutations, permute_data_nodes_attrs, permute_op_nodes_attrs
from mo.pipeline.common import prepare_emit_ir
from mo.utils import class_registration
from mo.utils.cli_parser import get_meta_info
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg
from mo.graph.graph import check_empty_graph


def driver(argv: argparse.Namespace, model_file_name: str, output_model_name: str, outputs: list, output_dir: str,
           scale: float,
           user_shapes: [None, list, np.array] = None,
           mean_scale_values: [dict, list] = ()):

    meta_info = get_meta_info(argv)

    model_proto = load_onnx_model(model_file_name)
    model_graph = model_proto.graph  # pylint: disable=no-member
    #print(model_graph)
    #assert len(model_graph) == 1, "An ONNX model contains more than 1 graph: unsupported"
    log.debug("Number of nodes in graph_def: {}".format(len(model_graph.node)))
    log.debug("Number of all input ports (not true inputs) in graph_def: {}".format(len(model_graph.input)))
    log.debug("Number of initializers in graph_def: {}".format(len(model_graph.initializer)))
    log.debug("Number of real inputs in graph_def: {}".format(len(model_graph.input) - len(model_graph.initializer)))
    update_extractors_with_extensions(onnx_op_extractors)

    try:
        graph = protobuf2nx(model_proto)
        log.debug("Number of nodes in NX graph: {}".format(graph.number_of_nodes()))
        graph.__setattr__('name', output_model_name if output_model_name else model_proto.graph.name)  # pylint: disable=no-member
        graph.graph['layout'] = 'NCHW'
        graph.graph['cmd_params'] = argv
        graph.graph['fw'] = 'onnx'
        graph.graph['feature_dim'] = 1 if graph.graph['layout'] == 'NCHW' else 3
        graph.graph['ir_version'] = 2 if argv.generate_deprecated_IR_V2 else 3
        # extract basic attributes earlier to enable some passes that relies on them before full attribute
        # extractor is called
        extract_node_attrs(graph, lambda node: (True, common_onnx_fields(node)))
    except Exception as e:
        raise Error(
            'Cannot pre-process ONNX graph after reading from model file "{}". ' \
            'File is corrupt or has unsupported format. Details: {}. ' +
            refer_to_faq_msg(44),
            model_file_name,
            str(e)
        ) from e
    check_empty_graph(graph, 'protobuf2nx. It may happen due to problems with loaded model')
    packed_user_shapes, packed_outputs, _ = user_data_repack(graph, user_shapes, outputs, None)

    output_op_nodes = add_output_ops(graph, packed_outputs)
    input_op_nodes = add_input_ops(graph, packed_user_shapes, True)

    # this call of 'graph_clean_up' removes child nodes of outputs which is useful when custom output is specified
    graph_clean_up(graph)
    check_empty_graph(graph, 'add_output_ops and add_input_ops')
    extract_node_attrs(graph, lambda node: onnx_op_extractor(node, check_for_duplicates(onnx_op_extractors)))

    class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)

    create_tensor_nodes(graph)
    graph_clean_up(graph)

    override_placeholder_shapes(graph, packed_user_shapes)
    override_batch(graph, argv.batch)

    graph_clean_up(graph)
    remove_op_nodes(graph, {'op': 'Identity'})

    graph_clean_up(graph)

    remove_output_ops(graph)

    partial_infer(graph)
    graph_clean_up(graph)
    check_empty_graph(graph, 'partial_infer')

    input_op_nodes = add_input_ops(graph, packed_user_shapes, False)
    graph_clean_up(graph)
    check_empty_graph(graph, 'add_input_ops')
    #change_placeholders_types_to_FP32(graph)

    scale_input(graph, scale)
    add_mean_scale_values(graph, mean_scale_values)

    convert_dilated_convolution(graph)
    graph_clean_up(graph)

    graph_clean_up(graph)

    remove_op_nodes(graph, {'op': 'Identity'})
    remove_useless_split(graph)

    class_registration.apply_replacements(graph, class_registration.ClassType.MIDDLE_REPLACER)

    convert_gemm_to_fully_connected(graph)
    NormalizeFullyConnected().find_and_replace_pattern(graph)

    fuse_pad(graph)
    graph_clean_up(graph)

    # Mark nodes with attr 'can_be_fused': False to disable fusing for specified nodes
    mark_unfused_nodes(graph, argv.finegrain_fusing)

    # Converting FusedBatchNorm layer to Mul->Add->Mul->Add sequence
    # IE doesn't support BN with 4 inputs, so we have to split it to two ScaleShift
    convert_batch_norm(graph)
    graph_clean_up(graph)

    if not argv.disable_fusing:
        # Converting ScaleShift layer to Mul->Add
        convert_scale_shift_to_mul_add(graph)
        graph_clean_up(graph)

        # Fusing the sequences of Mul/Add operations
        fuse_mul_add_sequence(graph)
        graph_clean_up(graph)

        # Fusing linear operation to Convolution
        fuse_linear_ops(graph)
        graph_clean_up(graph)

    if not argv.disable_gfusing:
        grouped_convolutions_fusing(graph)
        graph_clean_up(graph)
        if not argv.disable_fusing:
            fuse_linear_ops(graph)
            graph_clean_up(graph)

    convert_muladd_to_scaleshift_or_power(graph)
    graph_clean_up(graph)

    convert_mul_add_to_power(graph)
    graph_clean_up(graph)

    convert_reshape(graph)
    convert_add_to_scaleshift(graph)  # scale = 1
    convert_mul_to_scaleshift(graph)  # biases = 0

    fuse_pad(graph)
    graph_clean_up(graph)

    if argv.reverse_input_channels:
        reverse_input_channels(graph)

    if argv.move_to_preprocess:
        move_scaleshift_to_preprocess(graph)
        graph_clean_up(graph)

    fuse_sequence_of_reshapes(graph)
    graph_clean_up(graph)

    pattern = EltwiseInputNormalize()
    pattern.find_and_replace_pattern(graph)

    merge_nodes_permutations(graph)
    permute_data_nodes_attrs(graph)
    permute_op_nodes_attrs(graph)

    class_registration.apply_replacements(graph, class_registration.ClassType.BACK_REPLACER)

    prepare_emit_ir(graph=graph, data_type=argv.data_type, output_dir=output_dir, output_model_name=output_model_name,
                    meta_info=meta_info)

    return 0
