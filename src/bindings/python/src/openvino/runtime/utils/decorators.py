# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from inspect import getfullargspec
from typing import Any, Callable, List

from openvino.runtime import Node, Output
from openvino.runtime.utils.types import NodeInput, as_node, as_nodes


def _set_node_friendly_name(node: Node, /, **kwargs: Any) -> Node:
    if "name" in kwargs:
        node.friendly_name = kwargs["name"]
    return node


def nameable_op(node_factory_function: Callable) -> Callable:
    """Set the name to the openvino operator returned by the wrapped function."""

    @wraps(node_factory_function)
    def wrapper(*args: Any, **kwargs: Any) -> Node:
        node = node_factory_function(*args, **kwargs)
        node = _set_node_friendly_name(node, **kwargs)
        return node

    return wrapper


def _apply_affix(node: Node, prefix: str = "", suffix: str = "") -> Node:
    node.friendly_name = prefix + node.friendly_name + suffix
    return node


def apply_affix_on(*node_names: Any) -> Callable:
    """Add prefix and/or suffix to all openvino names of operators defined as arguments."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Node:
            arg_names = getfullargspec(func).args
            arg_mapping = dict(zip(arg_names, args))
            for node_name in node_names:
                # Apply only on auto-generated nodes. Create such node and apply affixes.
                # Any Node instance supplied by the user is keeping the name as-is.
                if node_name in arg_mapping and not isinstance(arg_mapping[node_name], (Node, Output)):
                    arg_mapping[node_name] = _apply_affix(as_node(arg_mapping[node_name]),
                                                          prefix=kwargs.get("prefix", ""),
                                                          suffix=kwargs.get("suffix", ""),
                                                          )
            results = func(**arg_mapping, **kwargs)
            return results
        return wrapper
    return decorator


def unary_op(node_factory_function: Callable) -> Callable:
    """Convert the first input value to a Constant Node if a numeric value is detected."""

    @wraps(node_factory_function)
    def wrapper(input_value: NodeInput, *args: Any, **kwargs: Any) -> Node:
        input_node = as_node(input_value)
        node = node_factory_function(input_node, *args, **kwargs)
        node = _set_node_friendly_name(node, **kwargs)
        return node

    return wrapper


def binary_op(node_factory_function: Callable) -> Callable:
    """Convert the first two input values to Constant Nodes if numeric values are detected."""

    @wraps(node_factory_function)
    def wrapper(left: NodeInput, right: NodeInput, *args: Any, **kwargs: Any) -> Node:
        left, right = as_nodes(left, right)
        node = node_factory_function(left, right, *args, **kwargs)
        node = _set_node_friendly_name(node, **kwargs)
        return node

    return wrapper


def custom_preprocess_function(custom_function: Callable) -> Callable:
    """Convert Node returned from custom_function to Output."""

    @wraps(custom_function)
    def wrapper(node: Node) -> Output:
        return Output._from_node(custom_function(node))

    return wrapper
