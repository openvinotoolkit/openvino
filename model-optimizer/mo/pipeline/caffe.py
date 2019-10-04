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

from extensions.back.CreateConstNodes import CreateConstNodesReplacement
from extensions.back.FuseReshapesSequence import FuseReshapesSequence
from extensions.back.RemoveRedundantReshapes import RemoveRedundantReshapes
from mo.front.caffe import custom_layers_mapping, loader
from mo.front.caffe.extractor import caffe_type_extractors, caffe_extractor
from mo.front.common.register_custom_ops import update_extractors_with_extensions, check_for_duplicates
from mo.front.extractor import extract_node_attrs, remove_output_ops
from mo.middle.passes.conv import convert_add_or_mul_to_scaleshift
from mo.middle.passes.conv import convert_muladd_to_scaleshift, \
    convert_matmul_to_fully_connected, batch_norm_fuse
from mo.middle.passes.eliminate import graph_clean_up
from mo.middle.passes.eliminate import remove_const_ops
from mo.middle.passes.fusing.decomposition import convert_bn_to_mul_add, convert_scale_shift_to_mul_add
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.fusing.resnet_optimization import stride_optimization
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.passes.shape import reverse_input_channels, merge_nodes_permutations, permute_data_nodes_attrs, \
    permute_op_nodes_attrs
from mo.pipeline.common import prepare_emit_ir
from mo.utils import class_registration
from mo.utils.cli_parser import get_meta_info
from mo.utils.error import Error
from mo.utils.find_inputs import find_inputs
from mo.utils.logger import log_step
from mo.utils.utils import refer_to_faq_msg


def driver(argv: argparse.Namespace, proto_file_name: str, model_file_name: str, output_model_name: str,
           output_dir: str, caffe_proto_path: str, mean_file: str = "",
           mean_file_offsets: tuple = None, custom_layers_mapping_path:str = None):
    log_step(argv.steps, 'LOAD')
    meta_info = get_meta_info(argv)

    caffe_pb2 = loader.import_caffe_pb2(caffe_proto_path)

    proto, model = loader.load_caffe_proto_model(caffe_pb2, proto_file_name, model_file_name)

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
    graph.print_graph_stat()
    graph.check_empty_graph('load_caffe_proto_model')

    graph.__setattr__('proto_path', proto_file_name)
    graph.__setattr__('caffemodel_path', model_file_name)
    graph.__setattr__('name', getattr(proto, 'name', None) or output_model_name)
    graph.graph['layout'] = 'NCHW'
    graph.graph['cmd_params'] = argv
    graph.graph['fw'] = 'caffe'
    if graph.graph['cmd_params'].generate_experimental_IR_V10:
        version = 10
    else:
        version = 6
    graph.graph['ir_version'] = 2 if argv.generate_deprecated_IR_V2 else version

    custom_layers_map = custom_layers_mapping.load_layers_xml(custom_layers_mapping_path)
    custom_layers_mapping.update_extractors(
        caffe_type_extractors,
        custom_layers_map,
        argv.disable_omitting_optional if hasattr(argv, 'disable_omitting_optional') else False,
        argv.enable_flattening_nested_params if hasattr(argv, 'enable_flattening_nested_params') else False
    )
    extract_node_attrs(graph, lambda node: caffe_extractor(node, check_for_duplicates(caffe_type_extractors)))

    # --------------------------------- LOAD END ------------------------------------------------------
    log_step(argv.steps, 'FRONT')
    class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)
    log_step(argv.steps, 'MIDDLE')
    class_registration.apply_replacements(graph, class_registration.ClassType.MIDDLE_REPLACER)

    # Mark nodes with attr 'can_be_fused': False to disable fusing for specified nodes
    mark_unfused_nodes(graph, argv.finegrain_fusing)

    # need this pass even without fusing to convert scale with 2 inputs
    convert_scale_shift_to_mul_add(graph)
    graph_clean_up(graph)

    if not argv.disable_fusing:
        convert_bn_to_mul_add(graph)
        graph_clean_up(graph)

        fuse_mul_add_sequence(graph)
        graph_clean_up(graph)

        fuse_linear_ops(graph)
        graph_clean_up(graph)

    if not argv.disable_resnet_optimization:
        stride_optimization(graph)

    convert_muladd_to_scaleshift(graph)
    convert_matmul_to_fully_connected(graph)
    batch_norm_fuse(graph)
    convert_add_or_mul_to_scaleshift(graph)  # scale = 1
    graph_clean_up(graph)

    log.debug("After graph_cleanup")
    graph.print_graph_stat()

    if argv.reverse_input_channels:
        reverse_input_channels(graph)

    if argv.move_to_preprocess:
        move_scaleshift_to_preprocess(graph)
        graph_clean_up(graph)

    FuseReshapesSequence().find_and_replace_pattern(graph)
    RemoveRedundantReshapes().find_and_replace_pattern(graph)

    input_names = find_inputs(graph)
    mf = []
    try:
        if mean_file and len(original_shapes) == 1:
            mf = loader.parse_mean(mean_file, original_shapes[input_names[0]], mean_file_offsets, caffe_pb2)
        elif mean_file:
            raise Error('Mean file for topologies with multiple inputs is not supported. ' +
                        refer_to_faq_msg(9))
    except ValueError as e:
        raise Error('Cannot load or process mean file: value error {}. ' +
                    refer_to_faq_msg(10), str(e)) from e

    merge_nodes_permutations(graph)
    permute_data_nodes_attrs(graph)
    permute_op_nodes_attrs(graph)

    graph_clean_up(graph)
    log_step(argv.steps, 'BACK')
    class_registration.apply_replacements(graph, class_registration.ClassType.BACK_REPLACER)

    remove_const_ops(graph)
    CreateConstNodesReplacement().find_and_replace_pattern(graph)

    remove_output_ops(graph)
    log_step(argv.steps, 'EMIT')
    prepare_emit_ir(graph=graph, data_type=argv.data_type, output_dir=output_dir, output_model_name=output_model_name,
                    mean_data=mf,
                    input_names=input_names,
                    meta_info=meta_info)
    return 0
