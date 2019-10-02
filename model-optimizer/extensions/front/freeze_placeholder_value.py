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

import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import SUPPORTED_DATA_TYPES
from mo.ops.const import Const
from mo.utils.error import Error


class FreezePlaceholderValue(FrontReplacementSubgraph):
    """
    Replaces existing placeholder to Constant node with provided value. It takes value from freeze_placeholder as
    a string and casts it to actual node data type
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['freeze_placeholder'] is not None]

    def run_after(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def run_before(self):
        return []
    
    @staticmethod
    def pattern():
        return dict(
            nodes=[('placeholder', dict(kind='op', op='Parameter'))],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        ph = match['placeholder']
        if ph.name in graph.graph['freeze_placeholder']:
            name = ph.name
            if ph.has_and_set('data_type'):
                data_type = ph.data_type
            else:
                data_type = SUPPORTED_DATA_TYPES[graph.graph['cmd_params'].data_type][0]
            string_value = graph.graph['freeze_placeholder'][name]
            try:
                if data_type != np.bool:
                    value = np.array(string_value, dtype=data_type)
                elif data_type == np.bool and graph.graph['fw'] == 'tf':
                    from mo.front.tf.common import tf_data_type_cast
                    if isinstance(string_value, list):
                        casted_list = list()
                        for v in np.array(string_value):
                            casted_list.append(tf_data_type_cast[ph.data_type](v))
                        value = np.array(string_value, dtype=data_type)
                    else:
                        value = tf_data_type_cast[ph.data_type](string_value)
                else:
                    raise Error("Can not cast value {} to {} data_type".format(string_value, data_type))
            except:
                raise Error("Can not cast value {} to {} data_type".format(string_value, data_type))
            try:
                value = np.reshape(a=value, newshape=ph.shape)
            except:
                raise Error("Can not reshape value {} to shape {}".format(value, ph.shape))
            out_edges = list(graph.out_edges(ph.id, data=True))
            new_node = Const(graph).create_node(
                attrs={'value': value, 'data_type': type(value), 'name': name + '/const_placeholder',
                       'shape': ph.shape})
            graph.erase_node(ph)
            graph.add_edges_from([(new_node.id, v, attrs) for u, v, attrs in out_edges])
            log.info("Placeholder node \"{}\" was replaced with Const node \"{}\" with value \"{}\"".format(
                name, new_node.name, value))
