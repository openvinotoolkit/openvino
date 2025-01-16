# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ExtractImagePatches import ExtractImagePatches
from openvino.tools.mo.front.common.partial_infer.utils import convert_tf_padding_to_str
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_int_list


class ExtractImagePatchesExtractor(FrontExtractorOp):
    op = 'ExtractImagePatches'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'spatial_dims': int64_array([1, 2]),
            'sizes': tf_int_list(node.pb.attr['ksizes'].list),
            'strides': tf_int_list(node.pb.attr['strides'].list),
            'rates': tf_int_list(node.pb.attr['rates'].list),
            'auto_pad': convert_tf_padding_to_str(node.pb.attr['padding'].s.decode()),
        }

        ExtractImagePatches.update_node_stat(node, attrs)
        return cls.enabled
