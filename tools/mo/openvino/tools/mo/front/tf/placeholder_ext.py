# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor, tf_tensor_shape
from openvino.tools.mo.ops.op import PermuteAttrs
from openvino.tools.mo.ops.parameter import Parameter


class PlaceholderFrontExtractor(FrontExtractorOp):
    op = 'Placeholder'
    enabled = True

    @classmethod
    def extract(cls, node):
        shape = shape_array([])
        # Extract output shape from `shape` attribute
        extracted_shape = tf_tensor_shape(node.pb.attr["shape"].shape)
        if len(extracted_shape) != 0:
            shape = extracted_shape
        else:
            # Extract output shape from `_output_shapes` attribute if it is possible
            extracted_output_shapes = node.pb.attr["_output_shapes"].list.shape
            if len(extracted_output_shapes) == 1:   # check if attribute not empty
                extracted_output_shapes = tf_tensor_shape(extracted_output_shapes[0])

                # Check equality of extracted shapes. We know some cases then Placeholder operation has empty `shape`
                # attribute value and non-empty `_output_shapes` attribute value and need co handle and support it.
                if len(extracted_output_shapes) > len(extracted_shape):
                    log.warning('Extracted shapes for Placeholder operation {} have different lengths: `shape` {} and '
                                '`_output_shapes` {}. Please, check if model is consistent'.format(
                        node.pb.name, extracted_shape, extracted_output_shapes))
                    if len(extracted_output_shapes) != 0:
                        shape = extracted_output_shapes

        attrs = {
            'data_type': tf_dtype_extractor(node.pb.attr["dtype"].type),
            'shape': shape,
            'permute_attrs': PermuteAttrs().update_attrs(attrs=[('shape', 'output:0')])
        }
        if node.pb.attr["shape"].shape.unknown_rank:
            attrs['shape'] = None
        Parameter.update_node_stat(node, attrs)
        return cls.enabled
