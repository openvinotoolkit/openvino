# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from openvino.tools.mo.load.loader import Loader
from openvino.tools.mo.front.caffe import custom_layers_mapping, loader
from openvino.tools.mo.front.caffe.extractor import caffe_type_extractors, caffe_extractor
from openvino.tools.mo.front.common.register_custom_ops import update_extractors_with_extensions, check_for_duplicates
from openvino.tools.mo.front.extractor import extract_node_attrs
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.telemetry_utils import send_op_names_info, send_shapes_info
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class CaffeLoader(Loader):
    enabled = True

    def load(self, graph: Graph):
        argv = graph.graph['cmd_params']
        if argv.caffe_parser_path is None:
            argv.caffe_parser_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'front', 'caffe', 'proto')
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

        if argv.k is None:
            argv.k = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'front', 'caffe', 'CustomLayersMapping.xml')
        custom_layers_map = custom_layers_mapping.load_layers_xml(argv.k)
        custom_layers_mapping.update_extractors(
            caffe_type_extractors,
            custom_layers_map,
            argv.disable_omitting_optional if hasattr(argv, 'disable_omitting_optional') else False,
            argv.enable_flattening_nested_params if hasattr(argv, 'enable_flattening_nested_params') else False
        )
        extract_node_attrs(graph, lambda node: caffe_extractor(node, check_for_duplicates(caffe_type_extractors)))
        send_op_names_info('caffe', graph)
        send_shapes_info('caffe', graph)
