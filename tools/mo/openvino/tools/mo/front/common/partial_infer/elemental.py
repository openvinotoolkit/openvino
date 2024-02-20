# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def single_output_infer(node, shape_infer, value_infer=None):
    node.out_node(0).shape = shape_infer(node)

    if value_infer is not None and \
            'value' in node.in_node() and \
            node.in_node().value is not None:
        node.out_node(0).value = value_infer(node)


def copy_shape_infer(node, value_infer=None):
    """
    Sets output dimensions of node equal to input ones
    Args:
        node: graph node
    """
    single_output_infer(node, lambda n: n.in_port(0).data.get_shape(), value_infer)


def copy_value(node):
    return None if node.in_node().value is None else node.in_port(0).data.get_value()
