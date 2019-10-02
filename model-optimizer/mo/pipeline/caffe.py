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

from mo.front.caffe import custom_layers_mapping, loader
from mo.front.caffe.extractor import caffe_type_extractors, caffe_extractor
from mo.front.common.register_custom_ops import update_extractors_with_extensions, check_for_duplicates
from mo.front.extractor import extract_node_attrs
from mo.pipeline.common import prepare_emit_ir
from mo.utils import class_registration
from mo.utils.cli_parser import get_meta_info
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def driver(argv: argparse.Namespace, proto_file_name: str, model_file_name: str, output_model_name: str,
           output_dir: str, caffe_proto_path: str, custom_layers_mapping_path: str = None):
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

    graph.graph['original_shapes'] = original_shapes
    graph.graph['caffe_pb2'] = caffe_pb2

    custom_layers_map = custom_layers_mapping.load_layers_xml(custom_layers_mapping_path)
    custom_layers_mapping.update_extractors(
        caffe_type_extractors,
        custom_layers_map,
        argv.disable_omitting_optional if hasattr(argv, 'disable_omitting_optional') else False,
        argv.enable_flattening_nested_params if hasattr(argv, 'enable_flattening_nested_params') else False
    )
    extract_node_attrs(graph, lambda node: caffe_extractor(node, check_for_duplicates(caffe_type_extractors)))

    # --------------------------------- LOAD END ------------------------------------------------------

    class_registration.apply_replacements(graph, [
        class_registration.ClassType.FRONT_REPLACER,
        class_registration.ClassType.MIDDLE_REPLACER,
        class_registration.ClassType.BACK_REPLACER
    ])

    prepare_emit_ir(graph=graph, data_type=argv.data_type, output_dir=output_dir, output_model_name=output_model_name,
                    mean_data=graph.graph['mf'],
                    input_names=graph.graph['input_names'],
                    meta_info=meta_info)
    return 0
