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

import networkx as nx
import numpy as np

from mo.front.extractor import add_attrs_props
from mo.front.extractor import update_ie_fields
from mo.graph.graph import Node, unique_id
from mo.middle.pattern_match import apply_pattern


def l2_norm_to_norm_action(graph: nx.MultiDiGraph, match: dict):
    input_data_name = match['input'].node
    output_data_name = match['l2_normalize_data'].node

    if not match['maximum_y_data'].has_valid('value'):
        return 1
    if match['maximum_y_data'].value.shape != ():
        return 1
    y = match['maximum_y_data'].value

    normalize_id = unique_id(graph)
    graph.add_node(normalize_id,
                   **add_attrs_props(
                       dict(kind='op', precision="FP32", type='Normalize', name=str(unique_id(graph, 'normalize')),
                            op='Normalize', shape=None, eps=str(y), across_spatial=str(0), channel_shared=str(0),
                            data_type=None,
                            infer=None)))
    normalize_data_id = unique_id(graph)

    graph.add_node(normalize_data_id, **add_attrs_props(graph.node[output_data_name]))
    update_ie_fields(graph.node[normalize_id])
    weights_id = unique_id(graph, 'weights_')
    graph.add_node(weights_id, **add_attrs_props(
        dict(kind='data', precision="FP32", name=weights_id, value=None, shape=None, data_type=None, infer=None)))
    wnode = Node(graph, weights_id)
    wnode['value'] = np.ones(shape=match['input'].shape[-1],
                             dtype=match['input'].data_type)  # TODO feature dim instead of -1
    wnode['shape'] = np.array(wnode['value'].shape)
    output_edges = list(graph.out_edges(output_data_name, data=True))
    graph.remove_edges_from([
        (input_data_name, match['l2_normalize'].id),
        (input_data_name, match['square'].id)
    ])
    graph.remove_edges_from(list(graph.out_edges(output_data_name)))
    graph.remove_node(output_data_name)
    graph.add_edge(input_data_name, normalize_id, **{'in': 0})
    graph.add_edge(weights_id, normalize_id, **{'in': 1, 'bin': 'weights'})
    graph.add_edge(normalize_id, normalize_data_id, **{'out': 0})
    for data, owner, attr in output_edges:
        graph.add_edge(normalize_data_id, owner, **attr)


def l2_norm_to_norm(graph: nx.MultiDiGraph):
    apply_pattern(
        graph,
        nodes=[
            ('input', dict(kind='data')),
            ('l2_normalize', dict(kind='op', op='Mul')),
            ('l2_normalize_data', dict(kind='data')),
            ('maximum', dict(kind='op', op='Maximum')),
            ('maximum_data', dict(kind='data')),
            ('maximum_y_data', dict(kind='data')),
            ('rsqrt', dict(kind='op', op='Rsqrt')),
            ('rsqrt_data', dict(kind='data')),
            ('square', dict(kind='op', op='Square')),
            ('square_data', dict(kind='data')),
            ('sum', dict(kind='op', op='Sum')),
            ('sum_data', dict(kind='data')),
            ('range_data', dict(kind='data')),

        ],
        edges=[
            ('range_data', 'sum'),
            ('input', 'square'),
            ('square', 'square_data'),
            ('square_data', 'sum'),
            ('sum', 'sum_data'),
            ('maximum_y_data', 'maximum'),
            ('sum_data', 'maximum'),
            ('maximum', 'maximum_data'),
            ('maximum_data', 'rsqrt'),
            ('rsqrt', 'rsqrt_data'),
            ('rsqrt_data', 'l2_normalize'),
            ('input', 'l2_normalize'),
            ('l2_normalize', 'l2_normalize_data'),
        ],
        action=l2_norm_to_norm_action
    )
