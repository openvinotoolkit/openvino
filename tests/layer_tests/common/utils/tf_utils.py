# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def mix_array_with_value(input_array, value):
    input_shape = input_array.shape
    mask = np.random.randint(0, 2, input_shape).astype(bool)
    return np.where(mask, input_array, value)


def load_graph(model_file, output_nodes_for_freeze=None):
    is_meta = os.path.splitext(model_file)[-1] == ".meta"

    tf.compat.v1.reset_default_graph()
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef() if not is_meta else tf.compat.v1.MetaGraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    nodes_to_clear_device = graph_def.node if isinstance(graph_def, tf.compat.v1.GraphDef) else graph_def.graph_def.node
    for node in nodes_to_clear_device:
        node.device = ""

    if is_meta:
        with tf.compat.v1.Session() as sess:
            restorer = tf.compat.v1.train.import_meta_graph(graph_def)
            restorer.restore(sess, re.sub('\.meta$', '', model_file))
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph_def.graph_def,
                                                                               output_nodes_for_freeze)

    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    return graph


def collect_tf_references(model_path, feed_dict, out_layer, output_nodes_for_freeze=None):
    _feed_dict = dict()

    graph = load_graph(model_path, output_nodes_for_freeze)
    output_tensors_list = list()
    outputs_list = list()
    for input in feed_dict:
        input_node = [node for node in graph.as_graph_def().node if node.name == input][0]
        if input_node.op == "Placeholder":
            tensor = graph.get_tensor_by_name(input + ":0")
            _feed_dict[tensor] = feed_dict[input]
        else:
            for parrent_input in input_node.input:
                in_node = [node for node in graph.as_graph_def().node if node.name == parrent_input][0]
                if in_node.op in ['Const', 'Assign', 'NoOp', 'Assert']:
                    continue
                else:
                    tensor = graph.get_tensor_by_name(parrent_input + ":0")
                _feed_dict[tensor] = feed_dict[input]

    for output in out_layer:
        tensor = graph.get_tensor_by_name(output + ":0")
        output_tensors_list.append(tensor)
        outputs_list.append(output)
    with graph.as_default():
        with tf.compat.v1.Session(graph=graph) as sess:
            outputs = sess.run(output_tensors_list, feed_dict=_feed_dict)
    out_dict = dict(zip(outputs_list, outputs))
    return out_dict


def children(op, graph):
    op = graph.get_operation_by_name(op)
    return set(op for out in op.outputs for op in out.consumers())


def collect_control_dependencies(graph):
    control_dependents_map = {}
    for op in graph.get_operations():
        for control_input in op.control_inputs:
            if control_input.name not in control_dependents_map:
                control_dependents_map[control_input.name] = [op]
            else:
                control_dependents_map[control_input.name].append(op)
    return control_dependents_map


def summarize_graph(model_path, output_nodes_for_freeze=None, reshape_net=None):
    placeholders = dict()
    variables = list()
    outputs = list()
    graph = load_graph(model_path, output_nodes_for_freeze)
    unlikely_output_types = ['Const', 'Assign', 'NoOp', 'Placeholder', 'Assert', 'switch_t', 'switch_f',
                             'TensorArrayCloseV3']
    control_dependents_map = collect_control_dependencies(graph)
    for node in graph.as_graph_def().node:
        if node.op == 'Placeholder':
            node_dict = dict()
            node_dict['type'] = tf.DType(node.attr['dtype'].type).name
            node_dict['shape'] = str(node.attr['shape'].shape.dim).replace('\n', '').replace(' ', '').replace(
                'size:', '').replace('[', '').replace(']', '')
            node_dict['shape'] = tuple(map(lambda x: int(x) if x else 0, node_dict['shape'].split(',')))
            placeholders[node.name] = node_dict
        if node.op == "Variable" or node.op == "VariableV2":
            variables.append(node.name)
        if len(children(node.name, graph)) == 0 and node.name not in control_dependents_map:
            if node.op not in unlikely_output_types and node.name.split('/')[-1] not in unlikely_output_types:
                outputs.append(node.name)
    result = dict()
    result['inputs'] = placeholders
    result['outputs'] = outputs

    if reshape_net:
        out_layer = list(result['inputs'].keys()) + result['outputs']
        feed_dict = {}
        for inputl in reshape_net:
            feed_dict.update({inputl: np.ones(shape=reshape_net[inputl])})
        scoring_res = collect_tf_references(model_path=model_path, feed_dict=feed_dict, out_layer=out_layer)
        for layer in scoring_res:
            if layer in result['inputs']:
                result['inputs'][layer]['shape'] = scoring_res[layer].shape

    return result


def permute_axis(axis, permutation_inv):
    return permutation_inv[axis]


def save_to_pb(tf_model, path_to_saved_tf_model, model_name='model.pb'):
    tf.io.write_graph(tf_model, path_to_saved_tf_model, model_name, False)
    assert os.path.isfile(os.path.join(path_to_saved_tf_model, model_name)), "model.pb haven't been saved " \
                                                                             "here: {}".format(path_to_saved_tf_model)
    return os.path.join(path_to_saved_tf_model, model_name)
