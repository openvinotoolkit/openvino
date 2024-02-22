# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import networkx as nx

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class CheckForCycle(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from openvino.tools.mo.middle.TensorIteratorMerge import TensorIteratorMerge
        return [TensorIteratorMerge]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        is_acyclic = nx.is_directed_acyclic_graph(graph)
        if not is_acyclic:
            raise Error('Graph contains a cycle. Can not proceed. ' + refer_to_faq_msg(97))
