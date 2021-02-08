# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
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

from typing import Optional
import numpy as np

from ngraph.impl import Node
from ngraph.utils.decorators import nameable_op
from ngraph.utils.node_factory import NodeFactory
from ngraph.utils.types import (
    as_node,
    NodeInput,
)


def _get_node_factory(opset_version: Optional[str] = None) -> NodeFactory:
    """! Return NodeFactory configured to create operators from specified opset version."""
    if opset_version:
        return NodeFactory(opset_version)
    else:
        return NodeFactory()
