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

import argparse
import copy
import logging as log

import networkx as nx
import numpy as np
import tensorflow as tf

from extensions.middle.EltwiseInputNormalization import EltwiseInputNormalize
from extensions.middle.TensorIteratorConditionChecker import ConditionChecks
from mo.middle.pattern_match import for_each_sub_graph, for_graph_and_each_sub_graph_recursively
import mo.front.tf.custom_subgraph_call as csc
from extensions.front.freeze_placeholder_value import FreezePlaceholderValue
from extensions.middle.TensorIteratorBackEdge import BackEdgesMatching
from extensions.middle.TensorIteratorCondition import LoopConditionMatcher
from extensions.middle.TensorIteratorInput import SmartInputMatcher, SimpleInputMatcher
from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
from extensions.middle.TensorIteratorOutput import SmartOutputMatcher
from extensions.middle.TensorIterator_utils import DeleteSelect
from mo.front.common.custom_replacement_registry import CustomReplacementRegistry
from mo.front.common.find_unsupported_ops import find_unsupported_ops
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.extractor import restore_edges, add_output_ops, add_input_ops, \
    extract_node_attrs, create_tensor_nodes, remove_output_ops, user_data_repack, remove_control_dependency_inputs
from mo.front.tf.change_placeholder_type import change_placeholders_types_to_FP32
from mo.front.tf.extractor import get_tf_edges, common_tf_fields, tf_op_extractor, tf_op_extractors
from mo.front.tf.loader import load_tf_graph_def, protobuf2nx
from mo.front.tf.register_custom_ops import update_registration
from mo.front.tf.replacement import FrontReplacementFromConfigFileOp
from mo.graph.graph import check_empty_graph
from mo.middle.passes.conv import convert_add_to_scaleshift, convert_matmul_to_fully_connected, \
    convert_muladd_to_scaleshift_or_power, fuse_pad, transpose_fully_connected_weights, \
    convert_dilated_convolution, convert_mul_to_scaleshift, convert_nasnet
from mo.middle.passes.eliminate import remove_op_nodes, remove_useless_split, graph_clean_up_tf
from mo.middle.passes.fusing.decomposition import convert_batch_norm, convert_scale_shift_to_mul_add
from mo.middle.passes.fusing.fuse_grouped_conv import grouped_convolutions_fusing
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.infer import scale_input, override_placeholder_shapes, partial_infer, convert_mul_add_to_power, \
    update_fully_connected_shapes, add_mean_scale_values, override_batch, check_for_cycle, delete_not_executable, delete_control_flow_edges
from mo.middle.passes.l2normalization import l2_norm_to_norm
from mo.middle.passes.leaky_relu import convert_mul_eltwise_to_leaky_relu
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.passes.pool import mean_to_avgpool
from mo.middle.passes.shape import convert_squeeze, convert_reshape, reverse_input_channels, \
    conv_flatten_concat, fuse_sequence_of_reshapes, repack_fully_connected_weights_nhwc_to_nchw, \
    apply_nhwc_to_nchw_permutation, permute_data_nodes_attrs, permute_op_nodes_attrs, merge_nodes_permutations
from mo.middle.passes.shared_weights_duplication import duplicate_shared_weights
from mo.pipeline.common import prepare_emit_ir
from mo.utils import class_registration, tensorboard
from mo.utils.cli_parser import get_meta_info
from mo.utils.custom_replacement_config import update_custom_replacement_config_file
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def need_to_repeat_conversion(graph: nx.MultiDiGraph):
    """ Detects if another round of conversion is required for the entire graph.

        It traverses a given `graph` and all sub-graphs recursively and searches for
        'repeat_conversion' graph attribute. If at least one is found and its value is True,
        this function returns True.
    """
    result = False

    def check_for_repeat(graph: nx.MultiDiGraph):
        if 'repeat_conversion' in graph.graph and graph.graph['repeat_conversion']:
            nonlocal result
            result = True

    for_graph_and_each_sub_graph_recursively(graph, check_for_repeat)

    return result


def tf2nx(argv: argparse.Namespace, model_file_name: str, output_model_name: str, outputs: list, output_dir: str,
          scale: float, is_binary: bool,
          user_shapes: [None, list, np.array] = None,
          mean_scale_values: [dict, list] = ()):
    """
    Convert TF GraphDef object to NetworkX representation.
    The resulting graph is still TF-specific and needs normalization passes to be applied.
    The specific TF structure assumes each GraphDef node is converted to a single
    NetworkX node, node id is an original TF node name, and edges go directly from one op   to another op.
    """
    meta_info = get_meta_info(argv)

    if argv.tensorflow_custom_layer_libraries:
        libraries = argv.tensorflow_custom_layer_libraries.split(',')
        for library in libraries:
            log.info('Loading library "{}" with custom operations'.format(library))
            tf.load_op_library(library)

    graph_def = load_tf_graph_def(graph_file_name=model_file_name, is_binary=is_binary,
                                  checkpoint=argv.input_checkpoint,
                                  user_output_node_names_list=outputs,
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
        graph.graph['ir_version'] = 2 if argv.generate_deprecated_IR_V2 else 3

        # placeholder for request from a transformation pass to repeat the entire conversion
        graph.graph['repeat_conversion'] = False

        graph = restore_edges(graph, get_tf_edges)
        graph = remove_control_dependency_inputs(graph)
        # extract basic attributes earlier to enable some passes that relies on them before full attribute
        # extractor is called
        extract_node_attrs(graph, lambda node: (True, common_tf_fields(node)))
    except Exception as e:
        raise Error(
            'Cannot pre-process TensorFlow graph after reading from model file "{}". ' \
            'File is corrupt or has unsupported format. Details: {}. ' +
            refer_to_faq_msg(44),
            model_file_name,
            str(e)
        ) from e

    check_empty_graph(graph, 'protobuf2nx. It may happen due to problems with loaded model')

    packed_user_shapes, packed_outputs, freeze_placeholder = user_data_repack(graph, user_shapes, outputs,
                                                                              argv.freeze_placeholder_with_value)
    if freeze_placeholder is not None:
        FreezePlaceholderValue.enabled = True
        FreezePlaceholderValue.replacement_dict = freeze_placeholder
        update_registration()

    inputs = list(packed_user_shapes.keys()) if packed_user_shapes is not None and isinstance(packed_user_shapes,
                                                                                              dict) else None
    graph.graph['inputs'] = inputs  # save user defined inputs for other extensions

    output_op_nodes = add_output_ops(graph, packed_outputs, inputs=packed_user_shapes)
    input_op_nodes = add_input_ops(graph, packed_user_shapes, True)

    # this call of 'graph_clean_up' removes child nodes of outputs which is useful when custom output is specified
    graph_clean_up_tf(graph)
    check_empty_graph(graph, 'add_output_ops and add_input_ops. It may happen due to absence of \'Placeholder\' layer '
                             'in the model')

    if argv.tensorflow_custom_operations_config_update:
        if update_custom_replacement_config_file(graph, argv.tensorflow_custom_operations_config_update):
            return 0
        else:
            return 1

    unsupported_ops_to_offload_to_tf = list()

    MAX_ITERATIONS = 5
    cur_iteration = 0
    while cur_iteration < MAX_ITERATIONS:
        graph_copy = copy.deepcopy(graph)  # create a copy of graph for the case when some ops are unsupported

        if argv.tensorflow_subgraph_patterns is not None:
            csc.replace_subgraph_calls(graph, argv.tensorflow_subgraph_patterns)

        if argv.tensorflow_operation_patterns is not None:
            csc.offload_operations_to_tf(graph, argv.tensorflow_operation_patterns)

        if argv.offload_unsupported_operations_to_tf and len(unsupported_ops_to_offload_to_tf):
            csc.offload_unsupported_operations_to_tf(graph, unsupported_ops_to_offload_to_tf)

        extract_node_attrs(graph, lambda node: tf_op_extractor(node, check_for_duplicates(tf_op_extractors)))

        if argv.tensorflow_use_custom_operations_config is not None:
            registry = CustomReplacementRegistry()
            registry.add_custom_replacement_description_from_config(argv.tensorflow_use_custom_operations_config)

            # automatically generate sub-classes for custom replacements that replace sub-graph with a single node
            for replacement_desc in registry.get_all_replacements_descriptions():
                if replacement_desc.has('op'):
                    type('FrontReplacementFromConfigFileOp' + replacement_desc.op, (FrontReplacementFromConfigFileOp,),
                         {'replacement_id': replacement_desc.id})
            update_registration()

        override_placeholder_shapes(graph, packed_user_shapes)

        # the user shapes are used to convert TensorFlow Object Detection API models
        graph.graph['user_shapes'] = packed_user_shapes
        class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)

        override_batch(graph, argv.batch)

        create_tensor_nodes(graph)
        graph_clean_up_tf(graph)

        remove_output_ops(graph)
        partial_infer(graph)
        delete_control_flow_edges(graph)

        # TENSOR ITERATOR CREATING BEGINS
        replacer = DeleteSelect()
        replacer.find_and_replace_pattern(graph)

        replacer = SmartInputMatcher()
        replacer.find_and_replace_pattern(graph)

        replacer = SmartOutputMatcher()
        replacer.find_and_replace_pattern(graph)

        replacer = LoopConditionMatcher()
        replacer.find_and_replace_pattern(graph)

        replacer = BackEdgesMatching()
        replacer.find_and_replace_pattern(graph)

        replacer = ConditionChecks()
        replacer.find_and_replace_pattern(graph)

        delete_not_executable(graph)
        graph_clean_up_tf(graph)

        replacer = SimpleInputMatcher()
        replacer.find_and_replace_pattern(graph)

        # Here will be optimizing path (ops afer Enter and before body take out of body)

        replacer = TensorIteratorMerge()
        replacer.find_and_replace_pattern(graph)

        # TENSOR ITERATOR CREATING ENDS

        check_for_cycle(graph)

        graph_clean_up_tf(graph)
        for_each_sub_graph(graph, graph_clean_up_tf)
        check_empty_graph(graph, 'partial_infer')

        csc.prepare_tf_call_nodes(graph)
        graph_clean_up_tf(graph)

        duplicate_shared_weights(graph)

        input_op_nodes = add_input_ops(graph, packed_user_shapes, False)
        graph_clean_up_tf(graph)
        check_empty_graph(graph, 'add_input_ops')

        change_placeholders_types_to_FP32(graph)

        scale_input(graph, scale)
        add_mean_scale_values(graph, mean_scale_values)

        convert_dilated_convolution(graph)
        graph_clean_up_tf(graph)
        for_each_sub_graph(graph, graph_clean_up_tf)

        l2_norm_to_norm(graph)
        graph_clean_up_tf(graph)

        remove_op_nodes(graph, {'op': 'Identity'})
        remove_op_nodes(graph, {'op': 'StopGradient'})
        remove_useless_split(graph)

        class_registration.apply_replacements(graph, class_registration.ClassType.MIDDLE_REPLACER)

        mean_to_avgpool(graph)
        convert_nasnet(graph)

        fuse_pad(graph)
        graph_clean_up_tf(graph)

        convert_matmul_to_fully_connected(graph)

        # Mark nodes with attr 'can_be_fused': False to disable fusing for specified nodes
        mark_unfused_nodes(graph, argv.finegrain_fusing)
        for_each_sub_graph(graph, lambda graph: mark_unfused_nodes(graph, argv.finegrain_fusing))

        # Converting FusedBatchNorm layer to Mul->Add->Mul->Add sequence
        # IE doesn't support BN with 4 inputs, so we have to split it to two ScaleShift
        convert_batch_norm(graph)
        graph_clean_up_tf(graph)

        if not argv.disable_fusing:
            # Converting ScaleShift layer to Mul->Add
            convert_scale_shift_to_mul_add(graph)
            graph_clean_up_tf(graph)

            # Fusing the sequences of Mul/Add operations
            fuse_mul_add_sequence(graph)
            for_each_sub_graph(graph, fuse_mul_add_sequence)
            graph_clean_up_tf(graph)
            for_each_sub_graph(graph, graph_clean_up_tf)

            # Fusing linear operation to Convolution
            fuse_linear_ops(graph)
            for_each_sub_graph(graph, fuse_linear_ops)
            graph_clean_up_tf(graph)
            for_each_sub_graph(graph, graph_clean_up_tf)

        if not argv.disable_gfusing:
            grouped_convolutions_fusing(graph)
            graph_clean_up_tf(graph)
            if not argv.disable_fusing:
                fuse_linear_ops(graph)
                graph_clean_up_tf(graph)

        # Converting Mul->Add to ScaleShift node
        convert_muladd_to_scaleshift_or_power(graph)
        graph_clean_up_tf(graph)

        convert_mul_add_to_power(graph)

        # Need to eliminate dead nodes before doing update_fully_connected_shapes
        # because update_fully_connected_shapes does partial inference and dead
        # nodes will lead to sporadic failures.
        graph_clean_up_tf(graph)
        update_fully_connected_shapes(graph)

        convert_mul_eltwise_to_leaky_relu(graph)
        graph_clean_up_tf(graph)

        fuse_pad(graph)
        graph_clean_up_tf(graph)

        convert_reshape(graph)
        convert_squeeze(graph)
        convert_add_to_scaleshift(graph)  # scale = 1
        convert_mul_to_scaleshift(graph)  # biases = 0

        if argv.reverse_input_channels:
            reverse_input_channels(graph)

        if argv.move_to_preprocess:
            move_scaleshift_to_preprocess(graph)
            graph_clean_up_tf(graph)

        fuse_sequence_of_reshapes(graph)

        pattern = EltwiseInputNormalize()
        pattern.find_and_replace_pattern(graph)

        conv_flatten_concat(graph)

        apply_nhwc_to_nchw_permutation(graph)
        for_each_sub_graph(graph, apply_nhwc_to_nchw_permutation)
        merge_nodes_permutations(graph)
        for_each_sub_graph(graph, merge_nodes_permutations)
        permute_data_nodes_attrs(graph)
        for_each_sub_graph(graph, permute_data_nodes_attrs)
        permute_op_nodes_attrs(graph)
        for_each_sub_graph(graph, permute_op_nodes_attrs)

        repack_fully_connected_weights_nhwc_to_nchw(graph)
        for_each_sub_graph(graph, repack_fully_connected_weights_nhwc_to_nchw)
        transpose_fully_connected_weights(graph)
        for_each_sub_graph(graph, transpose_fully_connected_weights)

        graph_clean_up_tf(graph)

        if argv.offload_unsupported_operations_to_tf:
            unsupported_ops_to_offload_to_tf = find_unsupported_ops(graph)
            if len(unsupported_ops_to_offload_to_tf) == 0:
                log.info('All operations are supported! Exit from the loop.')
                if not need_to_repeat_conversion(graph):
                    break
            else:
                print('After {} iteration there are {} unsupported ops'.format(cur_iteration + 1,
                                                                               len(unsupported_ops_to_offload_to_tf)))
        else:
            if not need_to_repeat_conversion(graph):
                break

        graph = graph_copy
        cur_iteration += 1

    class_registration.apply_replacements(graph, class_registration.ClassType.BACK_REPLACER)

    prepare_emit_ir(graph=graph, data_type=argv.data_type, output_dir=output_dir, output_model_name=output_model_name,
                    meta_info=meta_info)

    return 0
