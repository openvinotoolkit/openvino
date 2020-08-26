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

from extensions.ops.interpolate import Interpolate
from mo.front.common.replacement import FrontReplacementOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version
from mo.graph.graph import Graph, Node
from mo.utils.error import Error


class ResizeReplacer(FrontReplacementOp):
    op = 'Resize'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        mode = onnx_attr(node, 'mode', 's', default=b'nearest').decode()
        align_corners = onnx_attr(node, 'coordinate_transformation_mode', 's', default=b'').decode()
        align_corners = 1 if align_corners == 'align_corners' else 0

        idx = len(node.in_nodes()) - 1
        res = Interpolate(graph, dict(name=node.name)).create_node([node.in_node(0), node.in_node(idx)])
        Interpolate.update_node_stat(res, {'mode': mode, 'align_corners': align_corners})

        return [res.id]
