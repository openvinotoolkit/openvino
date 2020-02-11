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
from extensions.ops.ReduceOps import ReduceProd, ReduceAnd, ReduceMax, ReduceMean, ReduceSum
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node


class AllFrontExtractor(FrontExtractorOp):
    op = 'All'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        keep_dims = node.pb.attr['keep_dims'].b
        ReduceAnd.update_node_stat(node, {'keep_dims': keep_dims})
        return cls.enabled


class MaxFrontExtractor(FrontExtractorOp):
    op = 'Max'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceMax.update_node_stat(node, {'keep_dims': node.pb.attr['keep_dims'].b})
        return cls.enabled


class MeanExtractor(FrontExtractorOp):
    op = 'Mean'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceMean.update_node_stat(node, {'keep_dims': node.pb.attr["keep_dims"].b})
        return cls.enabled


class ProdFrontExtractor(FrontExtractorOp):
    op = 'Prod'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceProd.update_node_stat(node, {'keep_dims': node.pb.attr["keep_dims"].b})
        return cls.enabled


class SumFrontExtractor(FrontExtractorOp):
    op = 'Sum'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceSum.update_node_stat(node, {'keep_dims': node.pb.attr["keep_dims"].b})
        return cls.enabled
