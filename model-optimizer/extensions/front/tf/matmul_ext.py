"""
 Copyright (c) 2019 Intel Corporation

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
from extensions.ops.MatMul import MatMul
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from mo.utils.error import Error


class MatMulExtractor(FrontExtractorOp):
    op = 'MatMul'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        unsupported_attrs = []
        for attr_name in ['adjoint_a', 'adjoint_b', 'a_is_sparse', 'b_is_sparse']:
            if attr_name in node.pb.attr and node.pb.attr[attr_name].b:
                unsupported_attrs.append(attr_name)
        if len(unsupported_attrs) != 0:
            raise Error('MatMul operation {} use unsupported attrs: {}'.format(node.id, unsupported_attrs))

        MatMul.update_node_stat(node,
                                {
                                    'transpose_a': node.pb.attr['transpose_a'].b,
                                    'transpose_b': node.pb.attr['transpose_b'].b,
                                })
        return cls.enabled
