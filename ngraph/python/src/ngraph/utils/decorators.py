# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from functools import wraps
from typing import Any, Callable

from ngraph.impl import Node
from ngraph.utils.types import NodeInput, as_node, as_nodes


def _set_node_friendly_name(node: Node, **kwargs: Any) -> Node:
    if "name" in kwargs:
        node.friendly_name = kwargs["name"]
    return node


def nameable_op(node_factory_function: Callable) -> Callable:
    """Set the name to the ngraph operator returned by the wrapped function."""

    @wraps(node_factory_function)
    def wrapper(*args: Any, **kwargs: Any) -> Node:
        node = node_factory_function(*args, **kwargs)
        node = _set_node_friendly_name(node, **kwargs)
        return node

    return wrapper


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
