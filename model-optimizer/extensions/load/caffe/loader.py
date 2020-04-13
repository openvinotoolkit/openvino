"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.load.loader import Loader
from mo.front.caffe import custom_layers_mapping, loader
from mo.front.caffe.extractor import caffe_type_extractors, caffe_extractor
from mo.front.common.register_custom_ops import update_extractors_with_extensions, check_for_duplicates
from mo.front.extractor import extract_node_attrs
from mo.graph.graph import Graph
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class CaffeLoader(Loader):
    enabled = True

    def load(self, graph: Graph):
        argv = graph.graph['cmd_params']
        caffe_pb2 = loader.import_caffe_pb2(argv.caffe_parser_path)

        proto, model = loader.load_caffe_proto_model(caffe_pb2, argv.input_proto, argv.input_model)

        update_extractors_with_extensions(
            caffe_type_extractors,
            argv.disable_omitting_optional if hasattr(argv, 'disable_omitting_optional') else False,
            argv.disable_flattening_optional_params if hasattr(argv, 'disable_flattening_optional_params') else False
        )

        try:
            original_shapes = loader.caffe_pb_to_nx(graph, proto, model)
        except ValueError as e:
            raise Error('Invalid prototxt file: value error {}. ' +
                        refer_to_faq_msg(11), str(e)) from e
        graph.check_empty_graph('load_caffe_proto_model')

        graph.__setattr__('proto_path', argv.input_proto)
        graph.__setattr__('caffemodel_path', argv.input_model)
        graph.__setattr__('name', getattr(proto, 'name', None) or argv.model_name)
        graph.graph['layout'] = 'NCHW'
        graph.graph['fw'] = 'caffe'
        graph.graph['original_shapes'] = original_shapes
        graph.graph['caffe_pb2'] = caffe_pb2

        custom_layers_map = custom_layers_mapping.load_layers_xml(argv.k)
        custom_layers_mapping.update_extractors(
            caffe_type_extractors,
            custom_layers_map,
            argv.disable_omitting_optional if hasattr(argv, 'disable_omitting_optional') else False,
            argv.enable_flattening_nested_params if hasattr(argv, 'enable_flattening_nested_params') else False
        )
        extract_node_attrs(graph, lambda node: caffe_extractor(node, check_for_duplicates(caffe_type_extractors)))
