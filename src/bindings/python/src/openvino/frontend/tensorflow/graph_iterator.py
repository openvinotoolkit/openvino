# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.frontend.tensorflow.py_tensorflow_frontend import _FrontEndPyGraphIterator as GraphIterator
from .node_decoder import TFGraphNodeDecoder
import tensorflow as tf


class GraphIteratorTFGraph(GraphIterator):
    def __init__(self, tf_graph, inner_graph=False, parent_graph=None, graph_inputs=None):
        GraphIterator.__init__(self)
        self.m_graph = tf_graph
        self.m_node_index = 0
        self.m_decoders = []
        self.m_inner_graph = inner_graph
        self.m_parent_graph = parent_graph
        self.m_inputs = graph_inputs

        # list(tf_graph.captures)[0][1].name # operations
        # list(tf_graph.captures)[0][0]._name # variables
        captures_dict = {}
        if hasattr(tf_graph, 'captures'):
            for var_tensor, op_tensor in tf_graph.captures:
                captures_dict[op_tensor.name] = var_tensor._name
        self.m_captures = captures_dict

        for op in tf_graph.get_operations():
            self.m_decoders.append(TFGraphNodeDecoder(op, inner_graph, self))

        self.m_iterators = {}
        self.m_functions = {}
        for func_name, func in self.m_graph._functions.items():
            # func_op = None
            # for op in tf_graph.get_operations():
            #     if op.type == 'StatefulPartitionedCall' or op.type == 'PartitionedCall':
            #         try:
            #             name = op.node_def.attr['f'].func.name
            #             if func_name == name:
            #                 func_op = op
            #                 break
            #         except:
            #             pass
            # if func_op is None:
            #     continue
            # TODO cash to ov.Model
            # self.m_iterators[func_name] = GraphIteratorTFGraph(func.graph, True, self, func_op.inputs)

            # self.m_iterators[func_name] = GraphIteratorTFGraph(func.graph, True, self, None)

            self.m_functions[func_name] = func.graph
            self.m_iterators[func_name] = None

    def get_input_names(self) -> list:
        # if hasattr(self.m_graph, 'inputs'):
        #     inputs = []
        #     for input in self.m_graph.inputs:
        #         if input.name not in self.m_captures or self.m_inner_graph:
        #             inputs.append(input.op.name)
        #     return inputs

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
        # if not hasattr(self.m_graph, 'outputs'):
        #     return outputs
        # ordered_outputs = [output.name for output in self.m_graph.outputs if output.name]
        # filtered_ordered_outputs = []
        # for output in ordered_outputs:
        #     if output in outputs:
        #         filtered_ordered_outputs.append(output)

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
        if self.m_iterators[func_name] is None:
            self.m_iterators[func_name] = GraphIteratorTFGraph(self.m_functions[func_name], True, self, None)
        return self.m_iterators[func_name]
