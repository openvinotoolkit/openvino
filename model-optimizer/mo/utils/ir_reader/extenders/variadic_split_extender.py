# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class VariadicSplit_extender(Extender):
    op = 'VariadicSplit'

    @staticmethod
    def extend(op: Node):
        op['out_ports_count'] = len(op.ports)
