# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender
from mo.utils.ir_reader.extenders.priorbox_extender import PriorBox_extender


class PriorBoxClustered_extender(Extender):
    op = 'PriorBoxClustered'

    @staticmethod
    def extend(op: Node):
        op['V10_infer'] = True

        PriorBox_extender.attr_restore(op, 'width', value=1.0)
        PriorBox_extender.attr_restore(op, 'height', value=1.0)
        PriorBox_extender.attr_restore(op, 'variance')
