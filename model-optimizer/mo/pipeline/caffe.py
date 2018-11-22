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
import logging as log

import numpy as np

from extensions.front.freeze_placeholder_value import FreezePlaceholderValue
from extensions.middle.FusePermutesSequence import FusePermutesSequence
from mo.front.caffe import custom_layers_mapping, loader
from mo.front.caffe.extractor import caffe_extractor, common_caffe_fields, caffe_type_extractors
from mo.front.common.register_custom_ops import check_for_duplicates
from mo.front.common.register_custom_ops import update_extractors_with_extensions
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.extractor import extract_node_attrs, add_output_ops, create_tensor_nodes, remove_output_ops, \
    add_input_ops, user_data_repack
from mo.graph.graph import print_graph_stat, check_empty_graph
from mo.middle.passes.conv import convert_muladd_to_scaleshift_or_power, \
    convert_matmul_to_fully_connected, batch_norm_fuse, convert_add_to_scaleshift, \
    convert_mul_to_scaleshift, \
    convert_multi_input_conv
from mo.middle.passes.eliminate import graph_clean_up, remove_op_nodes
from mo.middle.passes.fusing.decomposition import convert_bn_to_mul_add, convert_scale_shift_to_mul_add
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.fusing.resnet_optimization import stride_optimization
from mo.middle.passes.infer import add_mean_scale_values, scale_input, override_placeholder_shapes, mark_outputs, \
    partial_infer, convert_mul_add_to_power, override_batch
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.passes.pool import mean_to_avgpool
from mo.middle.passes.shape import reverse_input_channels, fuse_sequence_of_reshapes
from mo.middle.passes.shared_weights_duplication import duplicate_shared_weights
from mo.pipeline.common import prepare_emit_ir
from mo.utils import class_registration
from mo.utils.error import Error
from mo.utils.find_inputs import find_inputs
from mo.utils.utils import refer_to_faq_msg
from mo.utils.cli_parser import get_meta_info


def driver(argv: argparse.Namespace, proto_file_name: str, model_file_name: str, output_model_name: str, outputs: list,
           output_dir: str,
           scale: float,
           user_shapes: [None, list, np.array] = None, mean_scale_values: [dict, list] = (), mean_file: str = "",
           mean_file_offsets: tuple = None,
           custom_layers_mapping_path: str = None):
    meta_info = get_meta_info(argv)

    FusePermutesSequence.enabled = False

    try:
        proto, model = loader.load_caffe_proto_model(proto_file_name, model_file_name)
    except Error as e:
        raise
    except Exception as e:
        raise Error('Model Optimizer is not able to read {}. Possible reasons: '.format(proto_file_name) +
                    '1. your caffemodel contains custom layers that are not supported in Model Optimizer by default. ' +
                    '2. your prototxt does not have a valid structure, e.g you downloaded it as html. ' +
                    'In particular the first unknown field is {} '.format(str(e).split(' ')[-1]) +
                    'After you made sure that prototxt has a valid structure and still see this issue, then ' +
                    'you need to generate a python parser for caffe.proto that was used when the model ' +
                    'was created. ' +
                    'Run "python3 generate_caffe_pb2.py --input_proto ${PATH_TO_CAFFE}/src/caffe/proto/caffe.proto". ' +
                    refer_to_faq_msg(1)) from e

    update_extractors_with_extensions(
        caffe_type_extractors,
        argv.disable_omitting_optional if hasattr(argv, 'disable_omitting_optional') else False,
        argv.disable_flattening_optional_params if hasattr(argv, 'disable_flattening_optional_params') else False
    )

    try:
        graph, original_shapes = loader.caffe_pb_to_nx(proto, model)
    except ValueError as e:
        raise Error('Invalid prototxt file: value error {}. ' +
                    refer_to_faq_msg(11), str(e)) from e

    log.debug("After caffe_pb_to_nx")
    print_graph_stat(graph)
    check_empty_graph(graph, 'load_caffe_proto_model')

    graph.__setattr__('proto_path', proto_file_name)
    graph.__setattr__('caffemodel_path', model_file_name)
    graph.__setattr__('name', getattr(proto, 'name', None) or output_model_name)
    graph.graph['layout'] = 'NCHW'
    graph.graph['cmd_params'] = argv
    graph.graph['fw'] = 'caffe'
    graph.graph['ir_version'] = 2 if argv.generate_deprecated_IR_V2 else 3

    extract_node_attrs(graph, lambda node: (True, common_caffe_fields(node)))

    log.debug("After adding specific nodes for outputs")
    print_graph_stat(graph)

    custom_layers_map = custom_layers_mapping.load_layers_xml(custom_layers_mapping_path)
    custom_layers_mapping.update_extractors(
        caffe_type_extractors,
        custom_layers_map,
        argv.disable_omitting_optional if hasattr(argv, 'disable_omitting_optional') else False,
        argv.enable_flattening_nested_params if hasattr(argv, 'enable_flattening_nested_params') else False
    )

    extract_node_attrs(graph, lambda node: caffe_extractor(node, check_for_duplicates(caffe_type_extractors)))

    log.debug("After extract_node_attr")
    print_graph_stat(graph)

    packed_user_shapes, packed_outputs, freeze_placeholder = user_data_repack(graph, user_shapes, outputs, argv.freeze_placeholder_with_value)
    if argv.freeze_placeholder_with_value is not None:
        FreezePlaceholderValue.enabled = True
        FreezePlaceholderValue.replacement_dict = freeze_placeholder
        class_registration.update_registration([FrontReplacementSubgraph])
    output_op_nodes = add_output_ops(graph, packed_outputs)
    input_op_nodes = add_input_ops(graph, packed_user_shapes, True)
    override_placeholder_shapes(graph, packed_user_shapes)
    override_batch(graph, argv.batch)
    graph_clean_up(graph)
    check_empty_graph(graph, 'add_output_ops and add_input_ops')
    class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)

    graph = create_tensor_nodes(graph)

    log.debug("After create_tensor_nodes")
    print_graph_stat(graph)

    remove_op_nodes(graph, {'op': 'Identity'})
    remove_output_ops(graph)
    graph_clean_up(graph)

    log.debug("After removing specific nodes for output")
    print_graph_stat(graph)

    # you need to pass required network outputs here
    # but we don't have a way yet, so just passing all discovered sinks
    mark_outputs(graph)
    graph_clean_up(graph)
    log.debug("After graph_cleanup")
    print_graph_stat(graph)

    graph = partial_infer(graph)
    log.debug("After partial_infer")
    print_graph_stat(graph)
    check_empty_graph(graph, 'partial_infer')
    duplicate_shared_weights(graph)

    input_op_nodes = add_input_ops(graph, packed_user_shapes, False)
    graph_clean_up(graph)
    check_empty_graph(graph, 'add_input_ops')
    scale_input(graph, scale)

    add_mean_scale_values(graph, mean_scale_values)

    log.debug("Split multi input convolutions")
    convert_multi_input_conv(graph)

    graph_clean_up(graph)
    log.debug("After graph_cleanup")
    print_graph_stat(graph)

    remove_op_nodes(graph, {'op': 'Dropout'})
    remove_op_nodes(graph, {'phase': 0})
    graph_clean_up(graph)

    class_registration.apply_replacements(graph, class_registration.ClassType.MIDDLE_REPLACER)

    mean_to_avgpool(graph)

    # Mark nodes with attr 'can_be_fused': False to disable fusing for specified nodes
    mark_unfused_nodes(graph, argv.finegrain_fusing)

    if not argv.disable_fusing:
        convert_bn_to_mul_add(graph)
        graph_clean_up(graph)

        convert_scale_shift_to_mul_add(graph)
        graph_clean_up(graph)

        fuse_mul_add_sequence(graph)
        graph_clean_up(graph)

        fuse_linear_ops(graph)
        graph_clean_up(graph)

    if not argv.disable_resnet_optimization:
        stride_optimization(graph)

    convert_muladd_to_scaleshift_or_power(graph)
    convert_matmul_to_fully_connected(graph)
    batch_norm_fuse(graph)
    convert_mul_add_to_power(graph)
    convert_add_to_scaleshift(graph)  # scale = 1
    convert_mul_to_scaleshift(graph)  # biases = 0

    graph_clean_up(graph)
    log.debug("After graph_cleanup")
    print_graph_stat(graph)

    if argv.reverse_input_channels:
        reverse_input_channels(graph)

    if argv.move_to_preprocess:
        move_scaleshift_to_preprocess(graph)
        graph_clean_up(graph)

    fuse_sequence_of_reshapes(graph)

    input_names = find_inputs(graph)
    mf = []
    try:
        if mean_file and len(original_shapes) == 1:
            mf = loader.parse_mean(mean_file, original_shapes[input_names[0]], mean_file_offsets)
        elif mean_file:
            raise Error('Mean file for topologies with multiple inputs is not supported. ' +
                        refer_to_faq_msg(9))
    except ValueError as e:
        raise Error('Cannot load or process mean file: value error {}. ' +
                    refer_to_faq_msg(10), str(e)) from e

    class_registration.apply_replacements(graph, class_registration.ClassType.BACK_REPLACER)

    prepare_emit_ir(graph=graph, data_type=argv.data_type, output_dir=output_dir, output_model_name=output_model_name,
                    mean_data=mf,
                    input_names=input_names,
                    meta_info=meta_info)
    return 0
