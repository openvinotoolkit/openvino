# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.load.loader import Loader
from openvino.tools.mo.front.common.register_custom_ops import update_extractors_with_extensions
from openvino.tools.mo.front.extractor import extract_node_attrs
from openvino.tools.mo.front.kaldi.extractor import kaldi_extractor, kaldi_type_extractors
from openvino.tools.mo.front.kaldi.loader.loader import load_kaldi_model
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.telemetry_utils import send_shapes_info, send_op_names_info
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class KaldiLoader(Loader):
    enabled = True

    def load(self, graph: Graph):
        argv = graph.graph['cmd_params']
        try:
            load_kaldi_model(graph, argv.input_model)
        except Exception as e:
            raise Error('Model Optimizer is not able to parse Kaldi model {}. '.format(argv.input_model) +
                        refer_to_faq_msg(91)) from e
        graph.check_empty_graph('load_kaldi_nnet_model')
        graph.graph['layout'] = 'NCHW'
        graph.graph['fw'] = 'kaldi'

        update_extractors_with_extensions(kaldi_type_extractors)
        extract_node_attrs(graph, lambda node: kaldi_extractor(node))

        send_op_names_info('kaldi', graph)
        send_shapes_info('kaldi', graph)
