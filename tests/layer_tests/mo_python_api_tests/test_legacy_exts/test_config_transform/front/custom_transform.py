# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.tf.replacement import FrontReplacementFromConfigFileGeneral
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.error import Error


class ConfigBasedTestReplacement(FrontReplacementFromConfigFileGeneral):
    replacement_id = 'ConfigBasedTestReplacement'
    run_not_recursively = True

    def transform_graph(self, graph: Graph, replacement_descriptions):
        sigmoid_nodes = graph.get_op_nodes(op='Sigmoid')
        assert len(sigmoid_nodes) > 0, "Error while applying ConfigBasedTestReplacement."
