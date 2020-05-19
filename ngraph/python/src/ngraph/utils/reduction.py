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

from typing import Iterable, Optional

from ngraph.impl import Node


def get_reduction_axes(node: Node, reduction_axes: Optional[Iterable[int]]) -> Iterable[int]:
    """Get reduction axes if it is None and convert it to set if its type is different.

    If reduction_axes is None we default to reduce all axes.

    :param node: The node we fill reduction axes for.
    :param reduction_axes: The collection of indices of axes to reduce. May be None.
    :return: Set filled with indices of axes we want to reduce.
    """
    if reduction_axes is None:
        reduction_axes = set(range(len(node.shape)))

    if type(reduction_axes) is not set:
        reduction_axes = set(reduction_axes)
    return reduction_axes
