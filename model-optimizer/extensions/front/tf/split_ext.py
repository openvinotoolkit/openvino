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

from extensions.ops.split import VariadicSplit, Split, AttributedSplit
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node


class SplitVExtractor(FrontExtractorOp):
    op = 'SplitV'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        VariadicSplit.update_node_stat(node, {'out_ports_count': node.pb.attr['num_split'].i,
                                              'swap_axis_and_split_size_inputs': True})
        return cls.enabled


class UnpackExtractor(FrontExtractorOp):
    op = 'Unpack'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        pb = node.pb
        AttributedSplit.update_node_stat(node,
                                         {
                                             'axis': pb.attr['axis'].i,
                                             'num_splits': pb.attr['num'].i,
                                             'squeeze_axis': True,
                                         })
        return cls.enabled


class SplitExtractor(FrontExtractorOp):
    op = 'Split'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        pb = node.pb
        Split.update_node_stat(node, {
            'num_splits': pb.attr['num_split'].i,
            'input_port': 1,
        })
        return cls.enabled
