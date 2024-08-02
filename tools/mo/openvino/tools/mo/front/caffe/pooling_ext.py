# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.caffe.extractors.utils import get_spatial_attr
from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.pooling import Pooling


class PoolingFrontExtractor(FrontExtractorOp):
    op = 'pooling'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.pooling_param

        method = 'max'
        exclude_pad = True
        kernel = [0, 0]
        stride = [1, 1]
        padding = [0, 0]
        global_pooling = False

        if hasattr(param, 'global_pooling') and param.global_pooling:
            global_pooling = param.global_pooling
        else:
            kernel = get_spatial_attr(kernel, 'kernel_size', 'kernel', param)
            padding = get_spatial_attr(padding, 'pad', 'pad', param)
            stride = get_spatial_attr(stride, 'stride', 'stride', param)

        if param.pool == 0:
            method = 'max'
            exclude_pad = True
        elif param.pool == 1:
            method = 'avg'
            exclude_pad = False
        else:
            raise ValueError('Unknown Pooling Method!')

        pooling_convention = 'full'  # for Caffe rounding type should be ceil
        rt = 'ceil'

        if hasattr(param, 'ceil_mode') and not param.ceil_mode:
            # If pooling has ceil_mode and ceil_mode is False using floor for rounding shapes in partial_infer
            pooling_convention = 'valid'
            rt = 'floor'

        attrs = {
            'window': int64_array([1, 1, kernel[1], kernel[0]]),
            'stride': int64_array([1, 1, stride[1], stride[0]]),
            'pad': int64_array([[0, 0], [0, 0], [padding[1], padding[1]], [padding[0], padding[0]]]),
            'pad_spatial_shape': int64_array([[padding[1], padding[1]], [padding[0], padding[0]]]),
            'pool_method': method,
            'exclude_pad': exclude_pad,
            'global_pool': global_pooling,
            'output_spatial_shape': None,
            'rounding_type': rt
        }

        attrs.update(layout_attrs())
        attrs['pooling_convention'] = pooling_convention

        # update the attributes of the node
        Pooling.update_node_stat(node, attrs)
        return cls.enabled
