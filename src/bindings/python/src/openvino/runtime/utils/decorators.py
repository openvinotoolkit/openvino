# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from inspect import getfullargspec
from typing import Any, Callable, List, Optional

from openvino.runtime import Node, Output
from openvino.runtime.utils.types import NodeInput, as_node, as_nodes


def _get_name(**kwargs: Any) -> Node:
    if "name" in kwargs:
        return kwargs["name"]
    return None


def _set_node_friendly_name(node: Node, *, name: Optional[str] = None) -> Node:
    if name is not None:
        node.friendly_name = name
    return node


def nameable_op(node_factory_function: Callable) -> Callable:
    """Set the name to the openvino operator returned by the wrapped function."""

    @wraps(node_factory_function)
    def wrapper(*args: Any, **kwargs: Any) -> Node:
        node = node_factory_function(*args, **kwargs)
        node = _set_node_friendly_name(node, name=_get_name(**kwargs))
        return node

    return wrapper


def unary_op(node_factory_function: Callable) -> Callable:
    """Convert the first input value to a Constant Node if a numeric value is detected."""

    @wraps(node_factory_function)
    def wrapper(input_value: NodeInput, *args: Any, **kwargs: Any) -> Node:
        input_node = as_node(input_value, name=_get_name(**kwargs))
        node = node_factory_function(input_node, *args, **kwargs)
        node = _set_node_friendly_name(node, name=_get_name(**kwargs))
        return node

    return wrapper


def binary_op(node_factory_function: Callable) -> Callable:
    """Convert the first two input values to Constant Nodes if numeric values are detected."""

    @wraps(node_factory_function)
    def wrapper(left: NodeInput, right: NodeInput, *args: Any, **kwargs: Any) -> Node:
        left, right = as_nodes(left, right, name=_get_name(**kwargs))
        node = node_factory_function(left, right, *args, **kwargs)
        node = _set_node_friendly_name(node, name=_get_name(**kwargs))
        return node

    return wrapper


def custom_preprocess_function(custom_function: Callable) -> Callable:
    """Convert Node returned from custom_function to Output."""

    @wraps(custom_function)
    def wrapper(node: Node) -> Output:
        return Output._from_node(custom_function(node))

    return wrapper
