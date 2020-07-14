"""
 Copyright (C) 2018-2020 Intel Corporation

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
from extensions.middle.LeakyReluPattern import LeakyReLU
from extensions.middle.pass_separator import PostMiddleStart
from mo.graph.graph import Graph
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.replacement import MiddleReplacementPattern
from mo.utils.error import Error
from mo.utils.find_inputs import find_inputs
from mo.utils.utils import refer_to_faq_msg


class Preprocessing(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [LeakyReLU]

    def run_before(self):
        return [PostMiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        argv = graph.graph['cmd_params']
        if argv.move_to_preprocess:
            move_scaleshift_to_preprocess(graph)


class CaffeMeanFileProcessing(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'caffe']

    def run_after(self):
        return [Preprocessing]

    def run_before(self):
        return [PostMiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        from mo.front.caffe import loader
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
