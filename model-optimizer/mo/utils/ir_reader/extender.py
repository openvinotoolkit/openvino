# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from mo.front.common.partial_infer.utils import int64_array
from mo.utils import class_registration
from mo.utils.graph import Node


class Extender(object):
    registered_ops = {}
    registered_cls = []
    # Add the derived class to excluded_classes if one should not be registered in registered_ops
    excluded_classes = []

    @staticmethod
    def extend(op: Node):
        pass

    @staticmethod
    def get_extender_class_by_name(name: str):
        return __class__.registered_ops[name]

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.IR_READER_EXTENDER

    @staticmethod
    def attr_to_list(node: Node, attribute: str):
        if not node.has_valid(attribute):
            log.warning('Attribute {} missed in node {} with type {}!'.format(attribute, node.soft_get('name'),
                                                                              node.soft_get('type')))
        elif not isinstance(node[attribute], list):
            node[attribute] = [node[attribute]]

    @staticmethod
    def const_shape_infer(node: Node):
        i = len(node.in_nodes())
        for num in node.out_nodes():
            node.out_node(num).shape = int64_array(node.ports[i][0])
            i += 1
