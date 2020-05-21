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
from extensions.ops.range import Range
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor
from mo.graph.graph import Node


class RangeFrontExtractor(FrontExtractorOp):
    op = 'Range'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Range.update_node_stat(node, {'dtype': tf_dtype_extractor(node.pb.attr['type'].type)})
        return cls.enabled

