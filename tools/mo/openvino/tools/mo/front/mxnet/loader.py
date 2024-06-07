# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging as log
import os

import mxnet as mx
import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.extractor import add_outputs_identity
from openvino.tools.mo.front.mxnet.extractor import common_mxnet_fields
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_node_edges, load_params, init_rnn_states, create_mxnet_edge
from openvino.tools.mo.front.mxnet.nd_to_params import build_params_file
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


def load_symbol_nodes(model_name, input_symbol: str = None, legacy_mxnet_model: bool = False):
    if input_symbol:
        json_name = input_symbol
        if legacy_mxnet_model:
            log.warning('If you use --input_symbol with legacy MXNet models be sure that symbol and param names ' +
                        'have correct format supported by MXNet')
    else:
        json_name = '%s-symbol.json' % model_name
        input_symbol = json_name

    if legacy_mxnet_model and (input_symbol == json_name):
        log.warning('For legacy MXNet models Model Optimizer does not support conversion of old MXNet models' +
                    '(trained with 1.0.0 version of MXNet and lower) with custom layers. ' +
                    refer_to_faq_msg(93))
        sym = mx.symbol.load(json_name)
        model_nodes = json.loads(sym.tojson())
    else:
        if os.path.isfile(json_name):
            with open(json_name, 'r') as fd:
                model_nodes = json.load(fd)
        else:
            raise Error('Specified input json {} does not exist. ' +
                        refer_to_faq_msg(84), json_name)

    return model_nodes['nodes']


def parse_input_model(input_model):
    path_wo_ext = '.'.join(input_model.split('.')[:-1])
    model_name_w_iter = path_wo_ext.split(os.sep)[-1]
    iteration_number = int(model_name_w_iter.split('-')[-1])
    model_name = '-'.join(path_wo_ext.split('-')[:-1])
    return model_name, iteration_number


def load_symbol_def(input_model_name, input_symbol, input_names: str = '', nd_prefix_name: str = '',
                    pretrained_model_name: str = '', legacy_mxnet_model: bool = False):
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

    model_nodes = load_symbol_nodes(model_name, input_symbol, legacy_mxnet_model)

    return model_nodes, model_params, model_name, iteration_number


def symbol_attrs(symbol_node):
    return {'symbol_dict': symbol_node}


def symbol2nx(graph, model_nodes, model_params, input_names: str = ''):
    if not input_names:
        input_names = ('data',)
    else:
        input_names = input_names.split(',')

    graph.inputs_order = input_names

    rnn_states = init_rnn_states(model_nodes)
    names_rnn_states = list(rnn_states.keys())

    # as mxnet contain input layers as index of layer, for correct set up edges, we need provide index of layer with name of  graph node
    index_node_keys = {}
    fw_name_map = {}
    for i, node in enumerate(model_nodes):
        if node['name'] in model_params._arg_params and node['name'] not in input_names:
            node['value'] = mo_array(model_params._arg_params[node['name']].asnumpy(), dtype=np.float32)
        elif node['name'] in model_params._aux_params and node['name'] not in input_names:
            node['value'] = mo_array(model_params._aux_params[node['name']].asnumpy(), dtype=np.float32)
        elif node['name'] in names_rnn_states:
            node['value'] = np.zeros(rnn_states[node['name']], dtype=np.float32)
        node_name = graph.unique_id(node['name'])
        graph.add_node(node_name, **symbol_attrs(node))
        if hasattr(graph, 'op_names_statistic') and 'op' in node:
            if node['op'] != 'null':
                graph.op_names_statistic[node['op']] += 1
        graph.node[node_name].update(common_mxnet_fields(Node(graph, node_name)))
        index_node_keys[i] = node_name
        fw_name_map[node_name] = node['name']

    used_indices_set = set()
    for i, attrs in enumerate(model_nodes):
        node = attrs
        edges, used_indices = get_mxnet_node_edges(node, i, list(model_nodes), index_node_keys)
        if len(edges) > 0:
            graph.add_edges_from(edges)
        used_indices_set = used_indices_set.union(used_indices)

    output_ids = [index_node_keys[node_id] for node_id in set(range(len(model_nodes))) - used_indices_set]

    graph.outputs_order = output_ids

    # Tensor names information corresponding to a node is stored on outgoing edges.
    # As output nodes do not have outgoing edges, fake outputs are required. In the following code
    # for each output Identity node is added, and tensor name for the output is kept
    # on (output, fake output) edge. After Result nodes adding transformation fake outputs
    # are deleted from graph.
    add_outputs_identity(graph, output_ids, lambda g, output_id, fake_node_id, fw_name: g.add_edges_from([
        create_mxnet_edge(output_id, fake_node_id, 0, 0, fw_name[output_id])]), {'fw_name': fw_name_map})

    return graph


def find_output_node(graph: Graph, src_input_index):
    for i, attrs in (list(graph.nodes(data=True))[src_input_index + 1:]):
        for input_index in attrs['symbol_dict']['inputs']:
            if input_index[0] == src_input_index:
                return i
