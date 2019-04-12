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

import os
import json

import numpy as np
import mxnet as mx
import logging as log

from mo.front.mxnet.extractors.utils import get_mxnet_node_edges, load_params, init_rnn_states
from mo.front.mxnet.extractor import common_mxnet_fields
from mo.front.mxnet.nd_to_params import build_params_file
from mo.graph.graph import Node, Graph
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def load_symbol_nodes(model_name, legacy_mxnet_model: bool = False):
    model_name = '%s-symbol.json' % model_name
    if legacy_mxnet_model:
        log.warning('For legacy MXNet models Model Optimizer does not support conversion of old MXNet models' +
                    '(trained with 1.0.0 version of MXNet and lower) with custom layers. ' +
                    refer_to_faq_msg(93))
        sym = mx.symbol.load(model_name)
        model_nodes = json.loads(sym.tojson())
    else:
        if os.path.isfile(model_name):
            model_nodes = json.load(open(model_name))
        else:
            raise Error('Specified input json {} does not exist. ' +
                        refer_to_faq_msg(84), model_name)

    return model_nodes['nodes']


def parse_input_model(input_model):
    path_wo_ext = '.'.join(input_model.split('.')[:-1])
    model_name_w_iter = path_wo_ext.split(os.sep)[-1]
    iteration_number = int(model_name_w_iter.split('-')[-1])
    model_name = '-'.join(path_wo_ext.split('-')[:-1])
    return model_name, iteration_number


def load_symbol_def(input_model_name, input_symbol, input_names: str = '', nd_prefix_name: str = '', pretrained_model_name: str = '', legacy_mxnet_model: bool = False):
    if not nd_prefix_name and not pretrained_model_name:
        # model name always has extension 'param'
        try:
            model_name, iteration_number = parse_input_model(input_model_name)
        except ValueError as err:
            raise Error(
                'Input model name {} is not in an expected format, cannot extract iteration number. ' +
                refer_to_faq_msg(48),
                input_model_name)

        if input_names:
            model_params = load_params(input_model_name, data_names=input_names.split(','))
        else:
            model_params = load_params(input_model_name)

    elif nd_prefix_name and pretrained_model_name and input_symbol:
        model_name, iteration_number = parse_input_model(pretrained_model_name)
        model_name = '-'.join(input_symbol.split('-')[:-1])
        model_params = build_params_file(nd_prefix_name, pretrained_model_name, input_names)
    else:
        raise Error(
            "Arguments --nd_prefix_name, --pretrained_model_name and --input_symbol should be provided. Please provide all or do not use any. " +
            refer_to_faq_msg(81))

    model_nodes = load_symbol_nodes(model_name, legacy_mxnet_model)

    return model_nodes, model_params, model_name, iteration_number


def symbol_attrs(symbol_node):
    return {'symbol_dict': symbol_node}


def symbol2nx(model_nodes, model_params, input_names: str = ''):
    if not input_names:
        input_names = ('data',)
    else:
        input_names = input_names.split(',')

    rnn_states = init_rnn_states(model_nodes)
    names_rnn_states = list(rnn_states.keys())

    graph = Graph()
    # as mxnet contain input layers as index of layer, for correct set up edges, we need provide index of layer with name of  graph node
    index_node_keys = {}
    for i, node in enumerate(model_nodes):
        if node['name'] in model_params._arg_params and node['name'] not in input_names:
            node['value'] = np.array(model_params._arg_params[node['name']].asnumpy(), dtype=np.float32)
        elif node['name'] in model_params._aux_params and node['name'] not in input_names:
            node['value'] = np.array(model_params._aux_params[node['name']].asnumpy(), dtype=np.float32)
        elif node['name'] in names_rnn_states:
            node['value'] = np.zeros(rnn_states[node['name']])
        node_name = graph.unique_id(node['name'])
        graph.add_node(node_name, **symbol_attrs(node))
        graph.node[node_name].update(common_mxnet_fields(Node(graph, node_name)))
        index_node_keys[i] = node_name

    for i, attrs in enumerate(model_nodes):
        node = attrs
        edges = get_mxnet_node_edges(node, i, list(model_nodes), index_node_keys)
        if len(edges) > 0:
            graph.add_edges_from(edges)

    return graph


def find_output_node(graph: Graph, src_input_index):
    for i, attrs in (list(graph.nodes(data=True))[src_input_index + 1:]):
        for input_index in attrs['symbol_dict']['inputs']:
            if input_index[0] == src_input_index:
                return i
