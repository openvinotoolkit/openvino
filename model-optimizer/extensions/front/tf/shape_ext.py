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
from mo.graph.graph import Node
from mo.ops.shape import Shape


class ShapeExtractor(FrontExtractorOp):
    op = 'Shape'
    enabled = True

    @staticmethod
    def extract(node: Node):
        Shape.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr['out_type'].type, np.int32)})
        return __class__.enabled
