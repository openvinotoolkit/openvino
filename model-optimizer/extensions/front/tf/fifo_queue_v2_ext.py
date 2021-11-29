# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor
from mo.ops.op import Op


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
                result_shapes.append(np.array([1, shape[0].size, shape[1].size, shape[2].size], dtype=np.int64))
            else:
                result_shapes.append(np.array([dim.size for dim in shape], dtype=np.int64))
        Op.update_node_stat(node, {'shapes': result_shapes, 'types': extracted_types})
        return cls.enabled
