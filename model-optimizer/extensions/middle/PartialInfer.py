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
        for param in graph.get_op_nodes(op='Parameter'):
            for dyn in [0, -1]:
                if dyn in param.soft_get('shape', []):
                    print('DYNAMIC INPUT DETECTED: MODEL={} SHAPE={}'.format(graph.graph['cmd_params'].model_name, param.shape))
                    exit(1)
        partial_infer(graph)
