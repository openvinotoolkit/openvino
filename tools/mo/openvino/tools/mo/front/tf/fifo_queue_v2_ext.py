# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor
from openvino.tools.mo.ops.op import Op


class FIFOQueueV2Extractor(FrontExtractorOp):
    op = 'FIFOQueueV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        shapes = node.pb.attr['shapes'].list.shape
        tf_types = node.pb.attr['component_types'].list.type
        extracted_types = []
        for t in tf_types:
            extracted_types.append(tf_dtype_extractor(t))
        result_shapes = []
        for shape_pb in shapes:
            shape = shape_pb.dim
            if len(shape) == 3:
                result_shapes.append(int64_array([1, shape[0].size, shape[1].size, shape[2].size]))
            else:
                result_shapes.append(int64_array([dim.size for dim in shape]))
        Op.update_node_stat(node, {'shapes': result_shapes, 'types': extracted_types})
        return cls.enabled
