# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.slice import CaffeSlice


class SliceFrontExtractor(FrontExtractorOp):
    op = 'slice'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.slice_param

        # slice_dim is deprecated parameter and is used as alias for axis
        # however if slice_dim is defined and axis is default, we use slice_dim
        if param.slice_dim != 1 and param.axis == 1:
            axis = param.slice_dim
        else:
            axis = param.axis

        update_attrs = {
            'axis': axis,
            'slice_point': int64_array(param.slice_point),
            'in_ports_count': 1,
            'out_ports_count': len(param.slice_point) + 1,
        }

        CaffeSlice.update_node_stat(node, update_attrs)
        return cls.enabled
