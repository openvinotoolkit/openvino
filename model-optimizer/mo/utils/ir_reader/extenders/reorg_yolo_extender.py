# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import int64_array
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class ReorgYolo_extender(Extender):
    op = 'ReorgYolo'

    @staticmethod
    def extend(op: Node):
        op['batch_dims'] = int64_array([0])
        op['channel_dims'] = int64_array([1])
        op['spatial_dims'] = [2, 3]
