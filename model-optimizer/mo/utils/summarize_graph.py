#!/usr/bin/env python3

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

import argparse
import os
import sys

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
unlikely_output_types = ['Const', 'Assign', 'NoOp', 'Parameter', 'Assert']


def children(op_name: str, graph: tf.Graph):
    op = graph.get_operation_by_name(op_name)
    return set(op for out in op.outputs for op in out.consumers())


def summarize_graph(graph_def):
    placeholders = dict()
    outputs = list()
    graph = tf.Graph()
    with graph.as_default():  # pylint: disable=not-context-manager
        tf.import_graph_def(graph_def, name='')
    for node in graph.as_graph_def().node:  # pylint: disable=no-member
        if node.op == 'Placeholder':
            node_dict = dict()
            node_dict['type'] = tf.DType(node.attr['dtype'].type).name
            node_dict['shape'] = str(tf.TensorShape(node.attr['shape'].shape)).replace(' ', '').replace('?', '-1')
            placeholders[node.name] = node_dict
        if len(children(node.name, graph)) == 0:
            if node.op not in unlikely_output_types and node.name.split('/')[-1] not in unlikely_output_types:
                outputs.append(node.name)
    result = dict()
    result['inputs'] = placeholders
    result['outputs'] = outputs
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    from mo.front.tf.loader import load_tf_graph_def

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, help="Path to tensorflow model", default="")
    parser.add_argument('--input_model_is_text', dest='text',
                        help='TensorFlow*: treat the input model file as a text protobuf format. If not specified, '
                             'the Model Optimizer treats it as a binary file by default.', action='store_true',
                        default=False)
    parser.add_argument('--input_meta', action='store_true',
                        help='TensorFlow*: treat the input model file as a meta graph def format', default=False)
    parser.add_argument("--input_checkpoint", type=str, help='TensorFlow variables file to load.', default="")
    parser.add_argument('--saved_model_dir', type=str, default="", help="TensorFlow saved_model_dir")
    parser.add_argument('--saved_model_tags', type=str, default="",
                        help="Group of tag(s) of the MetaGraphDef to load, in string \
                          format, separated by ','. For tag-set contains multiple tags, all tags must be passed in.")

    argv = parser.parse_args()
    if not argv.input_model and not argv.saved_model_dir:
        print("[ ERROR ] Please, provide --input_model and --input_model_is_text if needed or --input_dir for saved "
              "model directory")
        sys.exit(1)
    if argv.input_model and argv.saved_model_dir:
        print("[ ERROR ] Both keys were provided --input_model and --input_dir. Please, provide only one of them")
        sys.exit(1)
    graph_def, _ = load_tf_graph_def(graph_file_name=argv.input_model, is_binary=not argv.text,
                                     checkpoint=argv.input_checkpoint,
                                     model_dir=argv.saved_model_dir, saved_model_tags=argv.saved_model_tags)
    summary = summarize_graph(graph_def)
    print("{} input(s) detected:".format(len(summary['inputs'])))
    for input in summary['inputs']:
        print("Name: {}, type: {}, shape: {}".format(input, summary['inputs'][input]['type'],
                                                     summary['inputs'][input]['shape']))
    print("{} output(s) detected:".format(len(summary['outputs'])))
    print(*summary['outputs'], sep="\n")
