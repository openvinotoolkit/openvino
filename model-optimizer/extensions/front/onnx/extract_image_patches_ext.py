# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.ExtractImagePatches import ExtractImagePatches
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ExtractImagePatchesExtractor(FrontExtractorOp):
    op = 'ExtractImagePatches'
    enabled = True

    @classmethod
    def extract(cls, node):
        padding_mapping = {
            b'SAME': 'same_upper',
            b'VALID': 'valid'
        }

        attrs = {
            'spatial_dims': int64_array([1, 2]),
            'sizes': onnx_attr(node, 'ksizes', 'ints', default=None, dst_type=int64_array),
            'strides': onnx_attr(node, 'strides', 'ints', default=None, dst_type=int64_array),
            'rates': onnx_attr(node, 'rates', 'ints', default=None, dst_type=int64_array),
            'auto_pad': padding_mapping[onnx_attr(node, 'padding', 's', default=b"")],
        }

        ExtractImagePatches.update_node_stat(node, attrs)
