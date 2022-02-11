# Copyright (C) 2018-2022 Intel Corporation
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
        mf = []
        try:
            if argv.mean_file and len(original_shapes) == 1:
                mf = loader.parse_mean(argv.mean_file, original_shapes[input_names[0]],
                                       argv.mean_file_offsets, caffe_pb2)
            elif argv.mean_file:
                raise Error('Mean file for topologies with multiple inputs is not supported. ' +
                            refer_to_faq_msg(9))
        except ValueError as e:
            raise Error('Cannot load or process mean file: value error {}. ' +
                        refer_to_faq_msg(10), str(e)) from e

        graph.graph['mf'] = mf
        graph.graph['input_names'] = input_names
