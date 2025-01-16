# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class MemoryOffsetBatchUpdate(FrontReplacementPattern):
    """
    Update batch for MemoryOffset nodes with set element_size.
    element_size is set in loader according to shape saved in model (for example Parameter node have shape in attribute).
    But batch can be changed on front stage if user set batch through command line. So, element_size should be updated
    accordingly.
    """
    enabled = True
    run_not_recursively = True

    def run_after(self):
        from openvino.tools.mo.front.user_data_repack import UserDataRepack
        from openvino.tools.mo.front.kaldi.split_recurrent_memoryoffset import SplitRecurrentMemoryOffset
        return [UserDataRepack, SplitRecurrentMemoryOffset]

    def find_and_replace_pattern(self, graph: Graph):
        batch = graph.get_op_nodes(op="Parameter")[0].shape[0]
        for memoryoffset_node in graph.get_op_nodes(op='MemoryOffset'):
            if memoryoffset_node.has_valid('element_size'):
                memoryoffset_node.element_size[0] = batch
