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

import mxnet as mx

from mo.graph.graph import Node, Graph
from mo.ops.const import Const
from extensions.ops.elementwise import Elementwise
from mo.utils.error import Error
from mo.utils.str_to import StrTo
from mo.utils.utils import refer_to_faq_msg


class AttrDictionary(object):
    def __init__(self, dict):
        self._dict = dict

    def is_valid(self):
        return not self._dict is None

    def dict(self):
        return self._dict

    def add_dict(self, dict):
        self._dict.update(dict)

    def set(self, key, value):
        self._dict[key] = value

    def remove(self, key):
        if key in self._dict:
            del self._dict[key]

    def str(self, key, default=None):
        if not self.is_valid:
            if default is None:
                raise ValueError("Missing required parameter: " + key)
        if key in self._dict:
            return self._dict[key]
        return default

    def bool(self, key, default=None):
        attr = self.str(key, default)
        if isinstance(attr, str):
            if attr.isdigit():
                return bool(int(attr))
            return StrTo.bool(attr)
        else:
            return attr

    def float(self, key, default=None):
        return self.val(key, float, default)

    def int(self, key, default=None):
        return self.val(key, int, default)

    def tuple(self, key, valtype=str, default=None):
        attr = self.str(key, default)
        if attr is None:
            return default
        if isinstance(attr, str):
            if (not '(' in attr and not ')' in attr) and (not '[' in attr and not ']' in attr):
                return (valtype(attr),)
            if (not attr) or (not attr[1:-1].split(',')[0]):
                return tuple([valtype(x) for x in default])
            return StrTo.tuple(valtype, attr)
        else:
            return tuple([valtype(x) for x in attr])

    def list(self, key, valtype, default=None, sep=","):
        attr = self.str(key, default)
        if isinstance(attr, list):
            attr = [valtype(x) for x in attr]
            return attr
        else:
            return StrTo.list(attr, valtype, sep)

    def val(self, key, valtype, default=None):
        attr = self.str(key, default)
        attr = None if attr == 'None' else attr
        if valtype is None:
            return attr
        else:
            if not isinstance(attr, valtype) and attr is not None:
                return valtype(attr)
            else:
                return attr

    def has(self, key):
        if not self.is_valid:
            return False
        else:
            return key in self._dict


def get_mxnet_node_edges(node: dict, node_id: [int, str], nodes_list: list, index_node_key: dict):
    edge_list = []
    for in_port, src_node_id in enumerate(node['inputs']):
        src_node = src_node_id[0]
        dest_port = src_node_id[1]
        edge_attrs = {
            'in': in_port,
            'out': dest_port,
            # debug anchor for name of tensor consumed at this input port
            'fw_tensor_debug_info': [(nodes_list[src_node]['name'], src_node_id[1])],
            'in_attrs': ['in'],
            'out_attrs': ['out'],
            'data_attrs': ['fw_tensor_debug_info']
        }
        edge = (index_node_key[src_node], index_node_key[node_id], edge_attrs)
        edge_list.append(edge)
    return edge_list


def get_mxnet_layer_attrs(json_dic: dict):
    attr = 'param'
    if 'attr' in json_dic:
        attr = 'attr'
    elif 'attrs' in json_dic:
        attr = 'attrs'
    return AttrDictionary(json_dic[attr] if attr in json_dic else {})


def get_json_layer_attrs(json_dic):
    attr = 'param'
    if 'attr' in json_dic:
        attr = 'attr'
    elif 'attrs' in json_dic:
        attr = 'attrs'
    return json_dic[attr]


def load_params(input_model, data_names = ('data',)):
    arg_params = {}
    aux_params = {}
    arg_keys = []
    aux_keys = []
    file_format = input_model.split('.')[-1]
    loaded_weight = mx.nd.load(input_model)
    if file_format == 'params':
        for key in loaded_weight:
            keys = key.split(':')
            if len(keys)>1 and 'aux' == keys[0]:
                aux_keys.append(keys[1])
                aux_params[keys[1]] = loaded_weight[key]
            elif len(keys)>1 and 'arg' == keys[0]:
                arg_keys.append(keys[1])
                arg_params[keys[1]] = loaded_weight[key]
            else:
                arg_keys.append(key)
                arg_params[key] = loaded_weight[key]
    elif file_format == 'nd':
        for key in loaded_weight:
            if 'auxs' in input_model:
                aux_keys.append(key)
                aux_params[key] = loaded_weight[key]
            elif 'args' in input_model:
                arg_keys.append(key)
                arg_params[key] = loaded_weight[key]
    else:
        raise Error(
            'Unsupported Input model file type {}. Model Optimizer support only .params and .nd files format. ' +
            refer_to_faq_msg(85), file_format)

    data = mx.sym.Variable(data_names[0])
    model_params = mx.mod.Module(data, data_names=(data_names[0],), label_names=(data_names[0],))
    model_params._arg_params = arg_params
    model_params._aux_params = aux_params
    model_params._param_names = arg_keys
    model_params._aux_names = aux_keys
    return model_params


def init_rnn_states(model_nodes):
    states = {}
    for i, node in enumerate(model_nodes):
        if node['op'] == 'RNN':
            for i in node['inputs'][2:]:
                attrs = get_mxnet_layer_attrs(model_nodes[i[0]])
                shape = attrs.tuple('__shape__', int, None)
                if shape:
                    states.update({model_nodes[i[0]]['name']: shape})
    return states


def scalar_ops_replacer(graph: Graph, node: Node, elementwise_op_type=Elementwise):
    scalar_value = Const(graph, dict(value=node.scalar,
                                     symbol_dict={'name': node.id + '/const'})).create_node()
    lin_node = elementwise_op_type(graph, dict(name=node.id + '/lin_', symbol_dict={'name': node.id + '/lin_'})
                                   ).create_node()
    node.in_port(0).get_connection().set_destination(lin_node.in_port(0))
    lin_node.in_port(1).get_connection().set_source(scalar_value.out_port(0))
    node.out_port(0).get_connection().set_source(lin_node.out_port(0))
    return lin_node
