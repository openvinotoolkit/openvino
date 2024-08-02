# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging as log

from openvino.tools.mo.load.loader import Loader
from openvino.tools.mo.front.common.register_custom_ops import update_extractors_with_extensions, check_for_duplicates
from openvino.tools.mo.front.extractor import extract_node_attrs
from openvino.tools.mo.front.onnx.extractor import onnx_op_extractor, onnx_op_extractors
from openvino.tools.mo.front.onnx.loader import load_onnx_model, protobuf2nx
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.telemetry_utils import send_shapes_info, send_op_names_info
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class ONNXLoader(Loader):
    enabled = True
    run_not_recursively = True

    def load(self, graph: Graph):
        import onnx
        import io
        argv = graph.graph['cmd_params']
        if isinstance(argv.input_model, str):
            model_proto = load_onnx_model(argv.input_model)
        elif isinstance(argv.input_model, io.BytesIO):
            model_proto = onnx.load_model_from_string(argv.input_model.getvalue())
        else:
            raise Error('Unknown ONNX model type: {}'.format(type(argv.input_model)))

        model_graph = model_proto.graph  # pylint: disable=no-member
        # print(model_graph)
        # assert len(model_graph) == 1, "An ONNX model contains more than 1 graph: unsupported"
        log.debug("Number of nodes in graph_def: {}".format(len(model_graph.node)))
        log.debug("Number of all input ports (not true inputs) in graph_def: {}".format(len(model_graph.input)))
        log.debug("Number of initializers in graph_def: {}".format(len(model_graph.initializer)))
        log.debug(
            "Number of real inputs in graph_def: {}".format(len(model_graph.input) - len(model_graph.initializer)))
        update_extractors_with_extensions(onnx_op_extractors)

        try:
            protobuf2nx(graph, model_proto)
        except Exception as e:
            raise Error(
                'Cannot pre-process ONNX graph after reading from model file "{}". ' \
                'File is corrupt or has unsupported format. Details: {}. ' +
                refer_to_faq_msg(44),
                argv.input_model,
                str(e)
            ) from e
        log.debug("Number of nodes in NX graph: {}".format(graph.number_of_nodes()))

        graph.__setattr__('name',
                          argv.model_name if argv.model_name else model_proto.graph.name)  # pylint: disable=no-member
        graph.graph['layout'] = 'NCHW'
        graph.graph['fw'] = 'onnx'
        graph.graph['feature_dim'] = 1
        if hasattr(model_proto, 'opset_import'):
            graph.graph['fw_opset_version'] = model_proto.opset_import[0].version   # pylint: disable=no-member
        else:
            graph.graph['fw_opset_version'] = None

        graph.check_empty_graph('protobuf2nx. It may happen due to problems with loaded model')
        extract_node_attrs(graph, lambda node: onnx_op_extractor(node, check_for_duplicates(onnx_op_extractors)))
        send_op_names_info('onnx', graph)
        send_shapes_info('onnx', graph)
