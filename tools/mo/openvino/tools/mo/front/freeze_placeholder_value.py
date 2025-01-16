# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.convert_data_type import SUPPORTED_DATA_TYPES
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.utils.error import Error


class FreezePlaceholderValue(FrontReplacementSubgraph):
    """
    Replaces existing placeholder to Constant node with provided value. It takes value from freeze_placeholder as
    a string and casts it to actual node data type
    """
    enabled = True
    run_not_recursively = True
    graph_condition = [lambda graph: graph.graph['freeze_placeholder'] is not None]

    def run_after(self):
        from openvino.tools.mo.front.restore_ports import RestorePorts
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
                if data_type != bool:
                    value = mo_array(string_value, dtype=data_type)
                # TODO: investigate why boolean type is allowed only for TensorFlow
                elif data_type == bool and graph.graph['fw'] == 'tf':
                    from openvino.tools.mo.front.tf.common import tf_data_type_cast
                    if isinstance(string_value, list):
                        casted_list = list()
                        for v in mo_array(string_value):
                            casted_list.append(tf_data_type_cast[ph.data_type](v))
                        value = mo_array(string_value, dtype=data_type)
                    else:
                        value = tf_data_type_cast[ph.data_type](string_value)
                else:
                    raise Error("Cannot cast value {} to {} data_type".format(string_value, data_type))
            except:
                raise Error("Cannot cast value {} to {} data_type".format(string_value, data_type))
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
