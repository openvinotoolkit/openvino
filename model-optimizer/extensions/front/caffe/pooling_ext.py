"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import numpy as np

from mo.front.caffe.extractors.utils import get_spatial_attr
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp
from mo.ops.pooling import Pooling


class PoolingFrontExtractor(FrontExtractorOp):
    op = 'pooling'
    enabled = True

    @staticmethod
    def extract(node):
        proto_layer = node.pb
        param = proto_layer.pooling_param

        method = 'max'
        exclude_pad = 'true'
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
            exclude_pad = 'true'
        elif param.pool == 1:
            method = 'avg'
            exclude_pad = 'false'
        else:
            raise ValueError('Unknown Pooling Method!')

        pooling_convention = 'full'  # for Caffe rounding type should be ceil
        rt = 'ceil'

        if hasattr(param, 'ceil_mode') and not param.ceil_mode:
            # If pooling has ceil_mode and ceil_mode is False using floor for rounding shapes in partial_infer
            pooling_convention = 'valid'
            rt = 'floor'

        attrs = {
            'window': np.array([1, 1, kernel[1], kernel[0]], dtype=np.int64),
            'stride': np.array([1, 1, stride[1], stride[0]], dtype=np.int64),
            'pad': np.array([[0, 0], [0, 0], [padding[1], padding[1]], [padding[0], padding[0]]], dtype=np.int64),
            'pad_spatial_shape': np.array([[padding[1], padding[1]], [padding[0], padding[0]]], dtype=np.int64),
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
        return __class__.enabled
