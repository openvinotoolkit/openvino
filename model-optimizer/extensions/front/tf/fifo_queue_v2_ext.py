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

from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor
from mo.ops.op import Op


class FIFOQueueV2Extractor(FrontExtractorOp):
    op = 'FIFOQueueV2'
    enabled = True

    @staticmethod
    def extract(node):
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
        return __class__.enabled
