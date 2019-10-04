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
from extensions.back.CreateConstNodes import CreateConstNodesReplacement
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.utils.error import Error, FrameworkError
from mo.utils.logger import log_step
from mo.utils.utils import refer_to_faq_msg

try:
    import mxnet
except ImportError:
    raise Error('Module mxnet was not found. Please install appropriate version of mxnet via install_prerequisites '
                'script.' + refer_to_faq_msg(52))

import argparse

from mo.front.extractor import extract_node_attrs, remove_output_ops
from mo.front.mxnet.extractor import mxnet_op_extractor
from mo.front.mxnet.loader import symbol2nx, load_symbol_def
from mo.middle.passes.fusing.decomposition import convert_batch_norm, convert_scale_shift_to_mul_add
from mo.middle.passes.conv import convert_muladd_to_scaleshift, \
    convert_add_or_mul_to_scaleshift, fuse_pad, convert_matmul_to_fully_connected
from mo.middle.passes.eliminate import graph_clean_up, remove_const_ops
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.fusing.resnet_optimization import stride_optimization
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.passes.shape import reverse_input_channels, merge_nodes_permutations, permute_data_nodes_attrs, \
    permute_op_nodes_attrs
from mo.pipeline.common import prepare_emit_ir
from mo.front.mxnet.nd_to_params import save_params_file
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.mxnet.extractor import mxnet_op_extractors
from mo.utils import class_registration
from mo.utils.cli_parser import get_meta_info
from extensions.middle.EltwiseInputNormalization import EltwiseInputNormalize


def driver(argv: argparse.Namespace, input_model: str, output_model_name: str, output_dir: str):
    log_step(argv.steps, 'LOAD')
    meta_info = get_meta_info(argv)

    try:
        model_nodes, model_params, model_name, iteration_number = load_symbol_def(input_model, argv.input_symbol,
                                                                                  argv.input,
                                                                                  argv.nd_prefix_name,
                                                                                  argv.pretrained_model_name,
                                                                                  argv.legacy_mxnet_model)
    except (ValueError, mxnet.base.MXNetError) as e:
        raise FrameworkError(
            'The following error happened while loading mxnet model {}: {}. ' +
            refer_to_faq_msg(53),
            input_model,
            str(e)
        ) from e

    if argv.nd_prefix_name and argv.pretrained_model_name and argv.save_params_from_nd:
        save_params_file(model_name, model_params._arg_params, model_params._aux_params, iteration_number)

    update_extractors_with_extensions(mxnet_op_extractors)
    graph = symbol2nx(model_nodes, model_params, argv.input)
    graph.check_empty_graph('symbol2nx. It may happen due to problems with loaded model')

    graph.__setattr__('name', output_model_name)
    graph.graph['layout'] = 'NCHW'
    graph.graph['cmd_params'] = argv
    graph.graph['fw'] = 'mxnet'
    graph.graph['feature_dim'] = 1 if graph.graph['layout'] == 'NCHW' else 3

    if graph.graph['cmd_params'].generate_experimental_IR_V10:
        version = 10
    else:
        version = 6
    graph.graph['ir_version'] = 2 if argv.generate_deprecated_IR_V2 else version

    extract_node_attrs(graph, mxnet_op_extractor)

    # --------------------------------- LOAD END ------------------------------------------------------
    log_step(argv.steps, 'FRONT')
    class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)
    log_step(argv.steps, 'MIDDLE')
    class_registration.apply_replacements(graph, class_registration.ClassType.MIDDLE_REPLACER)

    fuse_pad(graph)

    # Mark nodes with attr 'can_be_fused': False to disable fusing for specified nodes
    mark_unfused_nodes(graph, argv.finegrain_fusing)

    # Converting FusedBatchNorm layer to Mul->Add->Mul->Add sequence
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

    if not argv.disable_resnet_optimization:
        stride_optimization(graph)

    fuse_pad(graph)

    # Converting Mul->Add to ScaleShift node
    convert_muladd_to_scaleshift(graph)
    graph_clean_up(graph)

    convert_add_or_mul_to_scaleshift(graph)  # scale = 1
    graph_clean_up(graph)

    if argv.reverse_input_channels:
        reverse_input_channels(graph)

    if argv.move_to_preprocess:
        move_scaleshift_to_preprocess(graph)
        graph_clean_up(graph)

    pattern = EltwiseInputNormalize()
    pattern.find_and_replace_pattern(graph)

    for_graph_and_each_sub_graph_recursively(graph, convert_matmul_to_fully_connected)

    merge_nodes_permutations(graph)
    permute_data_nodes_attrs(graph)
    permute_op_nodes_attrs(graph)

    graph_clean_up(graph)
    log_step(argv.steps, 'BACK')
    class_registration.apply_replacements(graph, class_registration.ClassType.BACK_REPLACER)

    for_graph_and_each_sub_graph_recursively(graph, remove_const_ops)
    CreateConstNodesReplacement().find_and_replace_pattern(graph)

    for_graph_and_each_sub_graph_recursively(graph, remove_output_ops)

    log_step(argv.steps, 'EMIT')
    prepare_emit_ir(graph=graph, data_type=argv.data_type, output_dir=output_dir, output_model_name=output_model_name,
                    meta_info=meta_info)
    return 0
