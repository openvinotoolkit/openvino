# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, shape_array, \
    dynamic_dimension_value, unmask_shape
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class PartialInfer(MiddleReplacementPattern):
    enabled = True
    run_not_recursively = True

    def run_after(self):
        from openvino.tools.mo.front.create_tensor_nodes import CreateTensorNodes
        return [CreateTensorNodes]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        dynamic_inputs = {}
        for parameter in graph.get_op_nodes(op='Parameter'):
            param_shape = parameter.soft_get('shape', shape_array(dynamic_dimension_value))
            if not is_fully_defined(param_shape):
                parameter_name = parameter.soft_get('name', parameter.id)
                dynamic_inputs[parameter_name] = param_shape
        partial_infer(graph)
