# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.strided_slice import StridedSlice


class MXSliceToStridedSliceReplacer(FrontReplacementOp):
    op = 'MXSlice'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']

        strided_slice_node = StridedSlice(graph, dict(name=node.id + '/strided_slice_',
                                                      shrink_axis_mask=np.zeros(len(node.crop_begin), dtype=np.int64),
                                                      new_axis_mask=np.zeros(len(node.crop_begin), dtype=np.int64),
                                                      ellipsis_mask=np.zeros(len(node.crop_begin), dtype=np.int64),
                                                      begin_mask=np.ones(len(node.crop_begin), dtype=np.int64),
                                                      end_mask=np.ones(len(node.crop_end), dtype=np.int64))).create_node()
        node.in_port(0).get_connection().set_destination(strided_slice_node.in_port(0))
        node.out_port(0).get_connection().set_source(strided_slice_node.out_port(0))

        crop_begin_node = Const(graph, dict(value=node.crop_begin,
                                            symbol_dict={'name': node.id + '/crop_begin_const'})).create_node()
        crop_end_node = Const(graph, dict(value=node.crop_end,
                                          symbol_dict={'name': node.id + '/crop_end_const'})).create_node()
        strided_slice_node.in_port(1).get_connection().set_source(crop_begin_node.out_port(0))
        strided_slice_node.in_port(2).get_connection().set_source(crop_end_node.out_port(0))

        if len(node.step) > 0:
            stride_node = Const(graph, dict(value=node.step,
                                            symbol_dict={'name': node.id + '/steps_const'})).create_node()
            strided_slice_node.in_port(3).get_connection().set_source(stride_node.out_port(0))
