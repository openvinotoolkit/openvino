# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender


class ReorgYolo_extender(Extender):
    op = 'ReorgYolo'

    @staticmethod
    def extend(op: Node):
        op['batch_dims'] = int64_array([0])
        op['channel_dims'] = int64_array([1])
        op['spatial_dims'] = [2, 3]
