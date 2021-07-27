# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Graph
from mo.middle.passes.infer import partial_infer
from mo.middle.replacement import MiddleReplacementPattern


class PartialInfer(MiddleReplacementPattern):
    enabled = True
    run_not_recursively = True

    def run_after(self):
        from extensions.front.create_tensor_nodes import CreateTensorNodes
        return [CreateTensorNodes]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        exit_with_error = False
        shapes_str = []
        for param in graph.get_op_nodes(op='Parameter'):
            if len(param.shape) == 0:
                exit_with_error = True
                shapes_str.append(param.id + '=?')
            for dyn in [0, -1]:
                if dyn in param.shape:
                    shapes_str.append(param.id + '{}'.format(param.shape))
                    exit_with_error = True
        if exit_with_error:
            print('DYNAMIC INPUT DETECTED: {} {} {}'.format(graph.graph['cmd_params'].model_name,
                                                            graph.graph['fw'],
                                                            ','.join(shapes_str)))
            exit(1)
        partial_infer(graph)
