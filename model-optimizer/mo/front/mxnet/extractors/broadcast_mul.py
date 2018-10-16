"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.utils import mark_input_bins
from mo.graph.graph import Node


def broadcast_mul_infer(node: Node):
    in_port = 0
    if node.in_node(1).value is None:
        in_port = 1
    weights_port = 1 - in_port
    node.out_node(0).shape = node.in_node(in_port).shape
    mark_input_bins(node, ['weights'], weights_port)


def broadcast_mul_ext(attrs):
    node_attrs = {
        'type': 'ScaleShift',
        'infer': broadcast_mul_infer
    }
    node_attrs.update(layout_attrs())
    return node_attrs
