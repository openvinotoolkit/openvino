"""
 Copyright (C) 2018-2020 Intel Corporation

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
