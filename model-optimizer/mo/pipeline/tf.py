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

import argparse
import logging as log

import tensorflow as tf

from extensions.back.CreateConstNodes import CreateConstNodesReplacement
from extensions.back.FuseReshapesSequence import FuseReshapesSequence
from extensions.back.InsertLayoutPropagationTransposes import InsertLayoutPropagationTranspose
from extensions.back.RemoveRedundantReshapes import RemoveRedundantReshapes
from extensions.middle.ConcatOptimization import ConcatOptimization
from extensions.middle.EltwiseChecker import EltwiseChecker
from extensions.middle.EltwiseInputNormalization import EltwiseInputNormalize
from extensions.middle.LayoutChangeForConstantShapePaths import LayoutChangeForConstantShapePaths
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.extractor import restore_edges, extract_node_attrs, remove_output_ops, remove_control_dependency_inputs
from mo.front.tf.extractor import get_tf_edges, tf_op_extractor, tf_op_extractors
from mo.front.tf.loader import load_tf_graph_def, protobuf2nx
from mo.middle.passes.conv import convert_add_or_mul_to_scaleshift, convert_matmul_to_fully_connected, \
    convert_muladd_to_scaleshift, fuse_pad
from mo.middle.passes.eliminate import graph_clean_up_tf
from mo.middle.passes.eliminate import remove_const_ops
from mo.middle.passes.fusing.decomposition import convert_batch_norm, convert_scale_shift_to_mul_add
from mo.middle.passes.fusing.fuse_grouped_conv import grouped_convolutions_fusing
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.infer import update_fully_connected_shapes
from mo.middle.passes.leaky_relu import convert_mul_eltwise_to_leaky_relu
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.passes.shape import reverse_input_channels, apply_nhwc_to_nchw_permutation, \
    permute_data_nodes_attrs, permute_op_nodes_attrs, merge_nodes_permutations, permute_input_data
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.pipeline.common import prepare_emit_ir
from mo.utils import class_registration, tensorboard
from mo.utils.cli_parser import get_meta_info
from mo.utils.error import Error
from mo.utils.logger import log_step
from mo.utils.utils import refer_to_faq_msg

try:
    import tensorflow.contrib
except:
    pass  # we try to import contrib for loading models that use contrib operations


def tf2nx(argv: argparse.Namespace, model_file_name: str, output_model_name: str, output_dir: str, is_binary: bool):
    """
    Convert TF GraphDef object to NetworkX representation.
    The resulting graph is still TF-specific and needs normalization passes to be applied.
    The specific TF structure assumes each GraphDef node is converted to a single
    NetworkX node, node id is an original TF node name, and edges go directly from one op   to another op.
    """
    log_step(argv.steps, 'LOAD')
    meta_info = get_meta_info(argv)

    if argv.tensorflow_custom_layer_libraries:
        libraries = argv.tensorflow_custom_layer_libraries.split(',')
        for library in libraries:
            log.info('Loading library "{}" with custom operations'.format(library))
            tf.load_op_library(library)

    graph_def, variables_values = load_tf_graph_def(graph_file_name=model_file_name, is_binary=is_binary,
                                                    checkpoint=argv.input_checkpoint,
                                                    user_output_node_names_list=argv.output,
                                                    model_dir=argv.saved_model_dir,
                                                    meta_graph_file=argv.input_meta_graph,
                                                    saved_model_tags=argv.saved_model_tags)

    try:
        tf.import_graph_def(graph_def, name='')
    except:
        log.warning("TensorFlow post-processing of loaded model was unsuccessful. "
                    "This is an optional step that Model Optimizer performs for any input model but it is not usually "
                    "required for all models."
                    "It likely means that the original model is ill-formed. "
                    "Model Optimizer will continue converting this model.")

    log.debug("Number of nodes in graph_def: {}".format(len(graph_def.node)))  # pylint: disable=no-member

    if argv.tensorboard_logdir:
        tensorboard.dump_for_tensorboard(graph_def, argv.tensorboard_logdir)

    update_extractors_with_extensions(tf_op_extractors)

    try:
        graph = protobuf2nx(graph_def)
        graph.__setattr__('name', output_model_name)
        # 'layout' parameter change may cause an issue in EltwiseInputReshape replacer
        # and convert_nhwc_to_nchw(graph)
        graph.graph['layout'] = 'NCHW' if argv.disable_nhwc_to_nchw else 'NHWC'
        graph.graph['cmd_params'] = argv
        graph.graph['fw'] = 'tf'

        if graph.graph['cmd_params'].generate_experimental_IR_V10:
            version = 10
        else:
            version = 6
        graph.graph['ir_version'] = 2 if argv.generate_deprecated_IR_V2 else version

        graph.graph['variables_values'] = variables_values
        del variables_values

        graph = restore_edges(graph, get_tf_edges)
        graph = remove_control_dependency_inputs(graph)
    except Exception as e:
        raise Error(
            'Cannot pre-process TensorFlow graph after reading from model file "{}". ' \
            'File is corrupt or has unsupported format. Details: {}. ' +
            refer_to_faq_msg(44),
            model_file_name,
            str(e)
        ) from e

    graph.check_empty_graph('protobuf2nx. It may happen due to problems with loaded model')
    extract_node_attrs(graph, lambda node: tf_op_extractor(node, check_for_duplicates(tf_op_extractors)))

    # --------------------------------- LOAD END ------------------------------------------------------
    log_step(argv.steps, 'FRONT')
    class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)
    log_step(argv.steps, 'MIDDLE')
    class_registration.apply_replacements(graph, class_registration.ClassType.MIDDLE_REPLACER)

    fuse_pad(graph)
    graph_clean_up_tf(graph)

    for_graph_and_each_sub_graph_recursively(graph, convert_matmul_to_fully_connected)

    # Mark nodes with attr 'can_be_fused': False to disable fusing for specified nodes
    for_graph_and_each_sub_graph_recursively(graph, lambda graph: mark_unfused_nodes(graph, argv.finegrain_fusing))

    # Converting FusedBatchNorm layer to Mul->Add->Mul->Add sequence
    # IE doesn't support BN with 4 inputs, so we have to split it to two ScaleShift
    convert_batch_norm(graph)
    graph_clean_up_tf(graph)

    if not argv.disable_fusing:
        # Converting ScaleShift layer to Mul->Add
        for_graph_and_each_sub_graph_recursively(graph, convert_scale_shift_to_mul_add)
        for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)

        # Fusing the sequences of Mul/Add operations
        for_graph_and_each_sub_graph_recursively(graph, fuse_mul_add_sequence)
        for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)

        # Fusing linear operation to Convolution
        for_graph_and_each_sub_graph_recursively(graph, fuse_linear_ops)
        for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)

    if not argv.disable_gfusing:
        for_graph_and_each_sub_graph_recursively(graph, grouped_convolutions_fusing)
        graph_clean_up_tf(graph)
        if not argv.disable_fusing:
            fuse_linear_ops(graph)
            graph_clean_up_tf(graph)

    for_graph_and_each_sub_graph_recursively(graph, EltwiseChecker().find_and_replace_pattern)

    # Converting Mul->Add to ScaleShift node
    for_graph_and_each_sub_graph_recursively(graph, convert_muladd_to_scaleshift)

    # Need to eliminate dead nodes before doing update_fully_connected_shapes
    # because update_fully_connected_shapes does partial inference and dead
    # nodes will lead to sporadic failures.
    for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)
    for_graph_and_each_sub_graph_recursively(graph, update_fully_connected_shapes)

    for_graph_and_each_sub_graph_recursively(graph, convert_mul_eltwise_to_leaky_relu)
    graph_clean_up_tf(graph)
    for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)

    for_graph_and_each_sub_graph_recursively(graph, fuse_pad)
    for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)

    for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)

    for_graph_and_each_sub_graph_recursively(graph, convert_add_or_mul_to_scaleshift)  # scale = 1
    for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)

    if argv.reverse_input_channels:
        reverse_input_channels(graph)

    if argv.move_to_preprocess:
        for_graph_and_each_sub_graph_recursively(graph, move_scaleshift_to_preprocess)
        graph_clean_up_tf(graph)

    FuseReshapesSequence().find_and_replace_pattern(graph)
    RemoveRedundantReshapes().find_and_replace_pattern(graph)

    EltwiseInputNormalize().find_and_replace_pattern(graph)

    if argv.enable_concat_optimization:
        ConcatOptimization().find_and_replace_pattern(graph)

    for_graph_and_each_sub_graph_recursively(graph, InsertLayoutPropagationTranspose().find_and_replace_pattern)

    LayoutChangeForConstantShapePaths().find_and_replace_pattern(graph)
    for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)

    for_graph_and_each_sub_graph_recursively(graph, apply_nhwc_to_nchw_permutation)
    for_graph_and_each_sub_graph_recursively(graph, merge_nodes_permutations)
    for_graph_and_each_sub_graph_recursively(graph, permute_data_nodes_attrs)
    for_graph_and_each_sub_graph_recursively(graph, permute_op_nodes_attrs)
    for_graph_and_each_sub_graph_recursively(graph, permute_input_data)

    for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)

    graph.graph['layout'] = 'NCHW'

    log_step(argv.steps, 'BACK')
    class_registration.apply_replacements(graph, class_registration.ClassType.BACK_REPLACER)
    for_graph_and_each_sub_graph_recursively(graph, graph_clean_up_tf)

    for_graph_and_each_sub_graph_recursively(graph, remove_const_ops)
    for_graph_and_each_sub_graph_recursively(graph, CreateConstNodesReplacement().find_and_replace_pattern)

    for_graph_and_each_sub_graph_recursively(graph, remove_output_ops)

    log_step(argv.steps, 'EMIT')
    prepare_emit_ir(graph=graph, data_type=argv.data_type, output_dir=output_dir, output_model_name=output_model_name,
                    meta_info=meta_info)

    return 0
