# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all ngraph ops."""
from functools import partial
from typing import Optional

import numpy as np
from ngraph.impl import Node
from ngraph.opset_utils import _get_node_factory
from ngraph.utils.decorators import nameable_op
from ngraph.utils.types import (
    NodeInput,
    as_nodes,
    as_node
)

_get_node_factory_opset9 = partial(_get_node_factory, "opset9")