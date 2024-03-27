# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender


class Convert_extender(Extender):
    op = 'Convert'

    @staticmethod
    def extend(op: Node):
        op['dst_type'] = destination_type_to_np_data_type(op.destination_type)
        # CompressQuantizeWeights generates IR with constant sub-graph, that should not be ConstFolded:
        #   Const(u8) -> Convert(to fp) -> (some eltwise operations) -> FakeQuantize
        if op.in_node().in_node().soft_get('type') == 'Const':
            op['stop_value_propagation'] = True
