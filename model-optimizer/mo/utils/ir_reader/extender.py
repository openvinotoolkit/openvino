"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
from mo.utils.graph import Node
from mo.utils import class_registration

from mo.front.common.partial_infer.utils import int64_array


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
            node.out_node(num).shape = int64_array(node.ports[i])
            i += 1
