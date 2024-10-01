# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.model_analysis import AnalyzeAction


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
    def iterator_get_next_analysis(cls, graph: Graph, inputs_desc: dict):
        message = None
        op_nodes = graph.get_op_nodes(op='IteratorGetNext')

        params = ''
        for iter_get_next in op_nodes:
            for port in iter_get_next.out_nodes().keys():
                inputs_desc['{}:{}'.format(iter_get_next.soft_get('name', iter_get_next.id), port)] = {
                    'shape': iter_get_next.shapes[port].tolist(),
                    'value': None,
                    'data_type': iter_get_next.types[port]
                }
                if params != '':
                    params = params + ','
                shape = str(iter_get_next.shapes[port].tolist()).replace(',', '')
                params = params + '{}:{}{}'.format(iter_get_next.soft_get('name', iter_get_next.id), port, shape)

        if len(op_nodes):
            message = 'It looks like there is IteratorGetNext as input\n' \
                      'Run the Model Optimizer without --input option \n' \
                      'Otherwise, try to run the Model Optimizer with:\n\t\t--input "{}"\n'.format(params)
        return message

    def analyze(self, graph: Graph):
        inputs_desc = dict()
        message = InputsAnalysis.iterator_get_next_analysis(graph, inputs_desc)
        inputs_to_ignore = InputsAnalysis.fifo_queue_analysis(graph, inputs_desc)
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
        return {'inputs': inputs_desc}, message
