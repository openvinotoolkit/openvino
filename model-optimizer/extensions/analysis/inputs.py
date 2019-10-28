"""
 Copyright (c) 2019 Intel Corporation

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
import logging as log

import numpy as np

from mo.graph.graph import Graph
from mo.utils.model_analysis import AnalyzeAction


class InputsAnalysis(AnalyzeAction):
    """
    The analyser gets information about model inputs and their default values if any.
    """

    @classmethod
    def fifo_queue_analysis(cls, graph: Graph, inputs_desc: dict):
        """
        The FIFOQueue with QueueDeque has a separate input that specifies the size of batch to extract from queue. This
        input is redundant and should be remove from the model analysis output.
        """
        inputs_to_ignore = set()
        for fifo_queue in graph.get_op_nodes(op='FIFOQueueV2'):
            if len(fifo_queue.get_outputs({'out': 0})) != 1:
                log.debug('The FIFOQueue operation "{}" has more than 1 consumers'.format(fifo_queue.id))
                continue
            queue_deque = fifo_queue.out_node(0)
            if queue_deque.op in ['QueueDequeueMany', 'QueueDequeueManyV2', 'QueueDequeueUpTo', 'QueueDequeueUpToV2']:
                queue_deque_input_1 = queue_deque.in_node(1)
                if queue_deque_input_1.op in ['Parameter', 'PlaceholderWithDefault']:
                    log.debug('Adding node "{}" to placeholder ignore list'.format(queue_deque_input_1.id))
                    inputs_to_ignore.add(queue_deque_input_1.id)

                # create input per each QueueDeque output port
                for port_ind in range(len(queue_deque.out_nodes())):
                    inputs_desc["{}:{}".format(queue_deque.id, port_ind)] = {'shape': fifo_queue.shapes[port_ind].tolist(),
                                                                             'value': None,
                                                                             'data_type': fifo_queue.types[port_ind]}
        return inputs_to_ignore

    @classmethod
    def ignore_mxnet_softmax_inputs(cls, graph: Graph):
        """
        MxNet Softmax layers may have additional inputs which should be ignored. Refer to the
        extensions/front/mxnet/check_softmax_node_inputs.py.
        """
        inputs_to_ignore = set()
        softmax_nodes = []
        [softmax_nodes.extend(graph.get_op_nodes(op=op)) for op in ('SoftMax', 'SoftmaxActivation', 'SoftmaxOutput')]
        for softmax_node in softmax_nodes:
            for i in range(1, len(softmax_node.in_nodes())):
                if softmax_node.in_node(i).has_valid('op') and softmax_node.in_node(i).op == 'Parameter':
                    inputs_to_ignore.add(softmax_node.in_node(i).id)
        return inputs_to_ignore

    def analyze(self, graph: Graph):
        inputs_desc = dict()

        inputs_to_ignore = InputsAnalysis.fifo_queue_analysis(graph, inputs_desc)
        if graph.graph['fw'] == 'mxnet':
            inputs_to_ignore.update(InputsAnalysis.ignore_mxnet_softmax_inputs(graph))

        inputs = graph.get_op_nodes(op='Parameter')
        for input in inputs:
            inputs_desc[input.name] = {'shape': input.soft_get('shape', None),
                                       'data_type': input.soft_get('data_type', None),
                                       'value': None,
                                       }

        placeholders_with_default = graph.get_op_nodes(op='PlaceholderWithDefault')
        for input in placeholders_with_default:
            inputs_desc[input.name] = {'shape': input.soft_get('shape', None),
                                       'data_type': input.soft_get('data_type', None),
                                       'value': input.in_node(0).value if 0 in input.in_nodes() and
                                                                          input.in_node(0).has_valid('value') else None}

        for input_to_ignore in inputs_to_ignore:
            del inputs_desc[input_to_ignore]

        # workaround for the ONNX models case where input shape is specified as string value like: "width", "height".
        # In this case the string value is converted to 0, but in fact it is an arbitrary value so should be -1
        if graph.graph['fw'] == 'onnx':
            for inp in inputs_desc.values():
                inp['shape'] = [-1 if item == 0 else item for item in inp['shape']]
        return {'inputs': inputs_desc}
