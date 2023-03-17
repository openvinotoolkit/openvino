# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.frontend.tensorflow.py_tensorflow_frontend import _FrontEndPyGraphIterator as GraphIterator
from .node_decoder import TFGraphNodeDecoder


class GraphIteratorTFGraph(GraphIterator):
    def __init__(self, tf_graph):
        GraphIterator.__init__(self)
        self.m_graph = tf_graph
        self.m_node_index = 0
        self.m_decoders = []

        # list(tf_graph.captures)[0][1].name # operations
        # list(tf_graph.captures)[0][0]._name # variables
        captures_dict = {}
        if hasattr(tf_graph, 'captures'):
            for var_tensor, op_tensor in tf_graph.captures:
                captures_dict[op_tensor.name] = var_tensor._name
        self.m_captures = captures_dict

        for op in tf_graph.get_operations():
            self.m_decoders.append(TFGraphNodeDecoder(op))

    def get_input_names(self) -> list:
        if hasattr(self.m_graph, 'inputs'):
            inputs = []
            for input in self.m_graph.inputs:
                if input.name not in self.m_captures:
                    inputs.append(input.name)
            return inputs

        inp_ops = filter(lambda op: op.type == "Placeholder" and len(op.inputs) == 0, self.m_graph.get_operations())
        inp_names = [inp.name for inp in inp_ops]
        return inp_names

    def get_output_names(self) -> list:
        if hasattr(self.m_graph, 'outputs'):
            return [output.name for output in self.m_graph.outputs]
        non_outputs = []
        for op in self.m_graph.get_operations():
            for inp in op.inputs:
                non_outputs.append(inp.op.name)

        outputs = []
        for op in self.m_graph.get_operations():
            if op.name not in non_outputs:
                for output in op.outputs:
                    outputs.append(output.name)
        return outputs

    def is_end(self) -> bool:
        return self.m_node_index >= len(self.m_decoders)

    def reset(self):
        self.m_node_index = 0

    def size(self) -> int:
        return len(self.m_decoders)

    def next(self):
        self.m_node_index += 1

    def get_decoder(self):
        return self.m_decoders[self.m_node_index]

    def get_body_graph_iterator(self, func_name):
        return GraphIteratorTFGraph(self.m_graph._functions[func_name].graph)
