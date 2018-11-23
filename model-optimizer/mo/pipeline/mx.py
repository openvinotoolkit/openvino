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
from mo.utils.error import Error, FrameworkError
from mo.utils.utils import refer_to_faq_msg

try:
    import mxnet
except ImportError:
    raise Error('Module mxnet was not found. Please install appropriate version of mxnet via install_prerequisites '
                'script.' + refer_to_faq_msg(52))

import logging as log

import numpy as np
import argparse
import networkx as nx

from mo.front.extractor import add_output_ops, extract_node_attrs, create_tensor_nodes, \
    add_input_ops, remove_output_ops, user_data_repack
from mo.front.mxnet.extractor import mxnet_op_extractor
from mo.front.mxnet.loader import symbol2nx, load_symbol_def
from mo.middle.passes.fusing.decomposition import convert_batch_norm, convert_scale_shift_to_mul_add
from mo.middle.passes.conv import convert_muladd_to_scaleshift_or_power, \
    convert_add_to_scaleshift, convert_mul_to_scaleshift, fuse_pad
from mo.middle.passes.eliminate import graph_clean_up, remove_op_nodes
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.shared_weights_duplication import duplicate_shared_weights
from mo.middle.passes.fusing.resnet_optimization import stride_optimization
from mo.middle.passes.infer import mark_outputs, override_placeholder_shapes, partial_infer, add_mean_scale_values, \
    scale_input, convert_mul_add_to_power
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.passes.shape import reverse_input_channels
from mo.pipeline.common import prepare_emit_ir
from mo.graph.graph import create_edge, Node, print_graph_stat, check_empty_graph
from mo.front.mxnet.nd_to_params import save_params_file
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.mxnet.extractor import mxnet_op_extractors
from mo.utils import class_registration
from mo.utils.cli_parser import get_meta_info
from extensions.middle.PadToPoolingMiddleReplacer import PadToPoolingMiddleReplacer
from extensions.middle.EltwiseInputNormalization import EltwiseInputNormalize


def add_input_data_to_prior_boxes(graph: nx.MultiDiGraph, input_names: str = ''):
    """
    PriorBox layer has data input unlike mxnet.
    Need to add data input to _contrib_MultiBoxPrior for
    for correct conversion to PriorBox layer.

    Parameters
    ----------
    graph : nx.MultiDiGraph
       Graph with loaded model.
    """
    if not input_names:
        input_names = ('data',)
    else:
        input_names = input_names.split(',')

    input_nodes = {}
    for node in graph.nodes():
        node = Node(graph, node)
        if node.has_valid('op') and node.name in input_names:
            input_nodes.update({node.id: node})

    if len(input_nodes) > 0:
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == '_contrib_MultiBoxPrior':
                create_edge(list(input_nodes.values())[0], node, out_port=0, in_port=1)


#TODO Remove the func after 'add_output_ops' will be moved to front replacer.
def check_softmax_node_inputs(graph: nx.MultiDiGraph):
    for i, attrs in list(graph.nodes(data=True)):
        if 'op' in attrs and attrs['op'] == 'Softmax':
            node = Node(graph, i)
            if len(node.in_nodes()) > 1:
                graph.remove_node(node.in_node(1).id)


def driver(argv: argparse.Namespace, input_model: str, output_model_name: str, outputs: list, output_dir: str,
           scale: float,
           placeholder_shapes: [None, list, np.array] = None,
           mean_scale_values: [dict, list] = ()):
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
    check_empty_graph(graph, 'symbol2nx. It may happen due to problems with loaded model')

    graph.__setattr__('name', output_model_name)
    graph.graph['layout'] = 'NCHW'
    graph.graph['cmd_params'] = argv
    graph.graph['fw'] = 'mxnet'
    graph.graph['ir_version'] = 2 if argv.generate_deprecated_IR_V2 else 3
    graph = extract_node_attrs(graph, mxnet_op_extractor)
    check_softmax_node_inputs(graph)

    user_shapes, packed_outputs, _ = user_data_repack(graph, placeholder_shapes, outputs, None)
    output_op_nodes = add_output_ops(graph, packed_outputs)
    input_op_nodes = add_input_ops(graph, user_shapes, True)

    try:
        override_placeholder_shapes(graph, user_shapes, argv.batch)
    except ValueError as err:
        raise Error(
            'The following error happened while processing input shapes: {}. ' +
            refer_to_faq_msg(54),
            str(err)
        ) from err
    check_empty_graph(graph, 'add_output_ops and add_input_ops')

    class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)
    add_input_data_to_prior_boxes(graph, argv.input)

    graph = create_tensor_nodes(graph)

    graph_clean_up(graph)
    remove_output_ops(graph)
    mark_outputs(graph)
    remove_output_ops(graph)

    graph_clean_up(graph)

    log.debug("After removing specific nodes for output")

    print_graph_stat(graph)

    graph = partial_infer(graph)
    graph_clean_up(graph)
    check_empty_graph(graph, 'partial_infer')

    duplicate_shared_weights(graph)

    scale_input(graph, scale)
    add_mean_scale_values(graph, mean_scale_values)

    remove_op_nodes(graph, {'op': 'Dropout'})
    remove_op_nodes(graph, {'op': '_copy'})

    graph_clean_up(graph)

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
    PadToPoolingMiddleReplacer().find_and_replace_pattern(graph)

    # Converting Mul->Add to ScaleShift node
    convert_muladd_to_scaleshift_or_power(graph)
    graph_clean_up(graph)

    convert_mul_add_to_power(graph)
    convert_add_to_scaleshift(graph)  # scale = 1
    convert_mul_to_scaleshift(graph)  # biases = 0

    if argv.reverse_input_channels:
        reverse_input_channels(graph)

    if argv.move_to_preprocess:
        move_scaleshift_to_preprocess(graph)
        graph_clean_up(graph)

    pattern = EltwiseInputNormalize()
    pattern.find_and_replace_pattern(graph)

    class_registration.apply_replacements(graph, class_registration.ClassType.BACK_REPLACER)

    prepare_emit_ir(graph=graph, data_type=argv.data_type, output_dir=output_dir, output_model_name=output_model_name,
                    meta_info=meta_info)
    return 0
