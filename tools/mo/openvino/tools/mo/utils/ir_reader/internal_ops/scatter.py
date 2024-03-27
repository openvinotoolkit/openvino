# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.scatter import ScatterUpdate, Scatter
from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value

class ScatterUpdateInternal(ScatterUpdate):
    @staticmethod
    def infer(node: Node):
        updates_value = node.in_port(2).data.get_value()
        if updates_value is not None and isinstance(updates_value, np.ma.masked_array) and updates_value.ndim == 1:
            # we need to normalize masked_array so that the value infer works as expected
            value = [item if item is not np.ma.masked else dynamic_dimension_value for item in updates_value]
            updates_value = np.ma.masked_equal(value, dynamic_dimension_value).astype(dtype=updates_value.dtype)
            node.in_port(2).data.set_value(updates_value)
        ScatterUpdate.infer(node)
