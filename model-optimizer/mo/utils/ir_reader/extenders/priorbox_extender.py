# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.multi_box_prior import multi_box_prior_infer_mxnet
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class PriorBox_extender(Extender):
    op = 'PriorBox'

    @staticmethod
    def extend(op: Node):
        op['V10_infer'] = True

        attrs = ['min_size', 'max_size', 'aspect_ratio', 'variance', 'fixed_ratio', 'fixed_size', 'density']
        for attr in attrs:
            PriorBox_extender.attr_restore(op, attr)

        if op.graph.graph['cmd_params'].framework == 'mxnet':
            op['infer'] = multi_box_prior_infer_mxnet
            op['stop_attr_upd'] = True

    @staticmethod
    def attr_restore(node: Node, attribute: str, value=None):
        # Function to restore some specific attr for PriorBox & PriorBoxClustered layers
        if not node.has_valid(attribute):
            node[attribute] = [] if value is None else [value]
        if isinstance(node[attribute], str):
            node[attribute] = []
        else:
            Extender.attr_to_list(node, attribute)
