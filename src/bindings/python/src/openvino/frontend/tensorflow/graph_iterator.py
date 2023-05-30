# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.frontend.tensorflow.py_tensorflow_frontend import _FrontEndPyGraphIterator as GraphIterator
from .node_decoder import TFGraphNodeDecoder
import tensorflow as tf


class GraphIteratorTFGraph(GraphIterator):
    def __init__(self, tf_graph, inner_graph=False):
        GraphIterator.__init__(self)
        self.m_graph = tf_graph
        self.m_node_index = 0
        self.m_decoders = []
        self.m_inner_graph = inner_graph

        for op in tf_graph.get_operations():
            self.m_decoders.append(TFGraphNodeDecoder(op, inner_graph))

        self.m_iterators = {}
        for func_name, func in self.m_graph._functions.items():
            self.m_iterators[func_name] = None

    def get_input_names(self) -> list:
        inp_ops = filter(lambda op: op.type == "Placeholder" and len(op.inputs) == 0, self.m_graph.get_operations())
        inp_names = []
        for input in inp_ops:
            if tf.dtypes.DType(input.node_def.attr['dtype'].type).name != 'resource' or self.m_inner_graph:
                inp_names.append(input.name)
        return inp_names

    def get_output_names(self) -> list:
        non_outputs = []
        for op in self.m_graph.get_operations():
            for inp in op.inputs:
                non_outputs.append(inp.op.name)

        outputs = []
        for op in self.m_graph.get_operations():
            if op.name not in non_outputs:
                for output in op.outputs:
                    outputs = [output.name] + outputs
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
        if func_name not in self.m_iterators:
            return None
        if self.m_iterators[func_name] is None:
            self.m_iterators[func_name] = GraphIteratorTFGraph(self.m_graph._functions[func_name].graph, True)
        return self.m_iterators[func_name]
