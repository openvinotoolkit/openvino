# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.middle.LeakyReluPattern import LeakyReLUFusion
from openvino.tools.mo.middle.pass_separator import PostMiddleStart
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.find_inputs import find_inputs
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class CaffeMeanFileProcessing(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'caffe']

    def run_after(self):
        return [LeakyReLUFusion]

    def run_before(self):
        return [PostMiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        from openvino.tools.mo.front.caffe import loader
        argv = graph.graph['cmd_params']
        original_shapes = graph.graph['original_shapes']
        caffe_pb2 = graph.graph['caffe_pb2']
        del graph.graph['caffe_pb2']
        input_names = find_inputs(graph)
        graph.graph['input_names'] = input_names
