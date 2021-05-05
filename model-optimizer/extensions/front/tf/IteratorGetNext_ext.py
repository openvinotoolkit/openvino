# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor
from mo.ops.op import Op


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
            shape = shape_pb.dim
            result_shapes.append(int64_array([dim.size for dim in shape]))
        Op.update_node_stat(node, {'shapes': result_shapes, 'types': extracted_types})
        return cls.enabled
