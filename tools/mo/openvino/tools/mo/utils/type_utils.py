# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.pipeline.common import convert_const_node_value_type
from openvino.tools.mo.utils.error import Error


def override_data_type_of_constant(node: Node, lhs_idx: int = 0, rhs_idx: int = 1):
    in_type_0 = node.in_port(lhs_idx).get_data_type()
    in_type_1 = node.in_port(rhs_idx).get_data_type()
    if in_type_0 != in_type_1:
        # in case of input values data type mismatch we try to change the type of the constant to match the type of
        # another input.
        in_node_0 = node.in_port(0).get_source().node
        in_node_1 = node.in_port(1).get_source().node

        if in_node_0.op == 'Const':
            node_to_convert, src_type, dst_type = in_node_0, in_type_0, in_type_1
        elif in_node_1.op == 'Const':
            node_to_convert, src_type, dst_type = in_node_1, in_type_1, in_type_0
        else:
            raise Error("{} operation '{}' has inputs of different data types: '{}' and '{}' "
                        "that cannot be aligned".format(node.soft_get('op'),
                                                        node.soft_get('name'),
                                                        in_type_0,
                                                        in_type_1))
        log.error("Changing Const node '{}' data type from {} to {} for {} operation".format(
            node_to_convert.soft_get('name', node_to_convert.id), src_type, dst_type, node.soft_get('op')),
            extra={'is_warning': True})
        convert_const_node_value_type(node_to_convert, dst_type)
