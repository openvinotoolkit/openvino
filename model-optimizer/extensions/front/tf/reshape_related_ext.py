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
import numpy as np

from extensions.ops.rank import Rank
from extensions.ops.size import Size
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_int_list, tf_dtype_extractor
from mo.graph.graph import Node
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape
from mo.ops.squeeze import Squeeze


class RankFrontExtractor(FrontExtractorOp):
    op = 'Rank'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Rank.update_node_stat(node)
        return cls.enabled


class ReshapeExtractor(FrontExtractorOp):
    op = 'Reshape'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Reshape.update_node_stat(node, {'special_zero': False})
        return cls.enabled


class ShapeExtractor(FrontExtractorOp):
    op = 'Shape'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Shape.update_node_stat(node, {'data_type': tf_dtype_extractor(node.pb.attr['out_type'].type, np.int32)})
        return cls.enabled


class SizeFrontExtractor(FrontExtractorOp):
    op = 'Size'
    enabled = True

    @classmethod
    def extract(cls, node):
        Size.update_node_stat(node)
        return cls.enabled


class SqueezeExtractor(FrontExtractorOp):
    op = 'Squeeze'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Squeeze.update_node_stat(node, {'squeeze_dims': tf_int_list(node.pb.attr['squeeze_dims'].list)})
        return cls.enabled
