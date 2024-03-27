# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils import class_registration
from openvino.tools.mo.utils.graph import Node


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
    def use_shapes_from_ir(node: Node):
        # This function used instead of operation shape inference function to set all output shapes the same as
        # restored from IR. Firstly, check equality of old (restored from IR) and
        # new (calculated while shape inference) input shapes
        node['new_input_shapes'] = list()
        for n in node.in_ports():
            # We use such condition to handle optional inputs
            if not node.in_port(n).disconnected():
                node.new_input_shapes.append(node.in_port(n).data.get_shape())
        assert len(node.new_input_shapes) == len(node.old_input_shapes), \
            'Something wrong happened while {} node with type {} copy shape inference! {} != {}'.format(
                node.name, node.type, len(node.new_input_shapes), len(node.old_input_shapes))
        for new_input_shape, old_input_shape in zip(node.new_input_shapes, node.old_input_shapes):
            assert np.array_equal(new_input_shape, old_input_shape), \
                'Something wrong happened while {} node with type {} copy shape inference! {} != {}'.format(
                    node.name, node.type, new_input_shape, old_input_shape)

        # We need to use number of connected input ports to avoid errors with numbering
        # in node.ports dictionary, where used numbers of input nodes
        connected_input_ports = []
        for n in node.in_ports():
            if not node.in_port(n).disconnected():
                connected_input_ports.append(node.in_port(n))
        i = len(connected_input_ports)

        # Set all output shapes the same as restored from IR
        for num in node.out_ports():
            if i in node.ports:
                node.out_port(num).data.set_shape(int64_array(node.ports[i][0]))
            else:
                assert node.out_port(num).data.get_shape() is not None, "Newly added port does not have set shape"
            i += 1
