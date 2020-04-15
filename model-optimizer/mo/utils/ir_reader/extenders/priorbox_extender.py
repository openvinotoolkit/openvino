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

from mo.front.common.partial_infer.multi_box_prior import multi_box_prior_infer_mxnet
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class PriorBox_extender(Extender):
    op = 'PriorBox'

    @staticmethod
    def extend(op: Node):
        op['V10_infer'] = True

        attrs = ['min_size', 'max_size', 'aspect_ratio']
        for attr in attrs:
            PriorBox_extender.attr_restore(op, attr)

        if op.graph.graph['cmd_params'].framework == 'mxnet':
            op['infer'] = multi_box_prior_infer_mxnet
            op['stop_attr_upd'] = True

    @staticmethod
    def attr_restore(node: Node, attribute: str, value=None):
        # Function to restore some specific attr for PriorBox & PriorBoxClustered layers
        if not node.has_valid(attribute):
            node[attribute] = [] if value is None else [value]
        if isinstance(node[attribute], str):
            node[attribute] = []
        else:
            Extender.attr_to_list(node, attribute)
