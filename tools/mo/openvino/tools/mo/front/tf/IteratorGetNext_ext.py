# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor, tf_tensor_shape
from openvino.tools.mo.ops.op import Op


class IteratorGetNextExtractor(FrontExtractorOp):
    op = 'IteratorGetNext'
    enabled = True

    @classmethod
    def extract(cls, node):
        shapes = node.pb.attr['output_shapes'].list.shape
        tf_types = node.pb.attr['output_types'].list.type
        extracted_types = []
        for t in tf_types:
            extracted_types.append(tf_dtype_extractor(t))
        result_shapes = []
        for shape_pb in shapes:
            result_shapes.append(tf_tensor_shape(shape_pb))
        Op.update_node_stat(node, {'shapes': result_shapes, 'types': extracted_types, 'out_ports_count': len(result_shapes)})
        return cls.enabled
