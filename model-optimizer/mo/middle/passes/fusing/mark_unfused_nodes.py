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

import logging as log
import re

from mo.graph.graph import Node, Graph
from mo.middle.passes.fusing.helpers import get_value_id


def _check_lin_op(node: Node, layout: str):
    lin_ops = ['Mul', 'Add']
    if node.soft_get('op') in lin_ops:
        weights_id = get_value_id(node)
        if weights_id is None:
            node.graph.node[node.id]['can_be_fused'] = False
            log.info('[ FUSING ] Node {} wasn\'t marked as fusable (no weights, probably this is element-wise operation'
                     ' that is not fusable)'.format(node.id))
            return

        node.graph.node[node.id]['can_be_fused'] = True
        log.info('[ FUSING ] Node {} marked as fusable'.format(node.id))


def mark_unfused_nodes(graph: Graph, regex_masks: str):
    regex_masks = [] if not regex_masks else regex_masks.split(',')
    nodes = graph.get_op_nodes()
    for node in nodes:
        if node.has_valid('can_be_fused'):
            continue
        disabled = False
        for mask in regex_masks:
            res = re.findall(mask, node.name)
            if res and len(res):
                graph.node[node.id]['can_be_fused'] = False
                log.info('[ FUSING ] Node {} wasn\'t marked as fusable (user decision {})'.format(node.id,mask))
                disabled = True
        if not disabled:
            _check_lin_op(node, graph.graph['layout'])


