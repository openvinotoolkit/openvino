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
import os
import re

from mo.utils.error import Error, FrameworkError
from mo.utils.utils import refer_to_faq_msg

try:
    import tensorflow as tf
except ImportError:
    raise Error('Module tensorflow was not found. Please install tensorflow 1.2 or higher. ' +
                refer_to_faq_msg(42))

from google.protobuf import text_format
from mo.graph.graph import create_graph_with_nodes, Graph
from mo.utils.summarize_graph import summarize_graph


def freeze_checkpoints(graph_def: tf.GraphDef, checkpoint_dir: str, output_node_names: list):
    """
    Loads all the variables in a graph and stores them in a separate dictionary. Freezes output nodes in the graph
    :param graph_def: GraphDef object holding the network.
    :param checkpoint_dir: path to directory with checkpoint files with values of graph variables.
    :param output_node_names: list of output node names.
    :return: GraphDef containing a simplified version of the original.
    """
    log.debug("Loading checkpoint files from directory: {}".format(checkpoint_dir))
    checkpoint_files = []
    for checkpoint_name in sorted(os.listdir(checkpoint_dir)):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.isfile(checkpoint_path):
            checkpoint_files.append(checkpoint_path)
            log.debug("File {} will be loaded".format(checkpoint_path))
        else:
            log.debug("Path {} is not a file. Skipping")

    if len(checkpoint_files) == 0:
        raise Error("There are no checkpoint files in directory: {}".format(checkpoint_dir))

    tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        uninitialized_variables = [str(v, 'utf-8') for v in set(sess.run(tf.report_uninitialized_variables()))]
        all_variables = [n.name for n in sess.graph.as_graph_def().node if n.op in ['Variable', 'VariableV2']]
        white_list = [v for v in all_variables if v not in uninitialized_variables]
        black_list = [v for v in all_variables if v in uninitialized_variables]
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, output_node_names,
                                                                        variable_names_whitelist=white_list,
                                                                        variable_names_blacklist=black_list)
    variable_values = {}
    for checkpoint_file in checkpoint_files:
        log.debug("Loading {}".format(checkpoint_file))
        with tf.Session() as sess:
            var_list = {}
            var_to_shape_map = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint_file).get_variable_to_shape_map()
            for key in var_to_shape_map:
                try:
                    tensor = sess.graph.get_operation_by_name(key).outputs[0]
                except KeyError:
                    continue
                var_list[key] = tensor
            tf.train.Saver(var_list=var_list).restore(sess, checkpoint_file)
            for name, tensor in var_list.items():
                variable_values[name] = sess.run(tensor)
    return output_graph_def, variable_values


def freeze_checkpoint(graph_def, checkpoint, output_node_names):
    """
    Replaces all the variables in a graph with constants of the same values.
    :param graph_def: GraphDef object holding the network.
    :param checkpoint: path to checkpoint file with values of variables.
    :param output_node_names: list of output node names
    :return: GraphDef containing a simplified version of the original.
    """
    tf.import_graph_def(graph_def, name="")

    with tf.Session() as sess:
        var_list = {}
        var_to_shape_map = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint).get_variable_to_shape_map()
        for key in var_to_shape_map:
            try:
                tensor = sess.graph.get_operation_by_name(key).outputs[0]
            except KeyError:
                continue
            var_list[key] = tensor
        tf.train.Saver(var_list=var_list).restore(sess, checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, output_node_names)
    return output_graph_def


def read_file_to_graph_def(graph_def: [tf.GraphDef, tf.MetaGraphDef], graph_file_name: str = "",
                           is_binary: bool = True):
    """
    Reads file to protobuf
    :param graph_def: GraphDef orr MetaGraphDef object to store the network
    :param graph_file_name: path to file with graph
    :param is_binary: flag to switch between binary and test protobuf format of graph file
    :return: GraphDef or MetaGaphDef containing the network with cleared device info.
    """
    try:
        if is_binary:
            with open(graph_file_name, "rb") as f:
                graph_def.ParseFromString(f.read())
        else:
            with open(graph_file_name, "r") as f:
                text_format.Merge(f.read(), graph_def)
        nodes_to_clear_device = graph_def.node if isinstance(graph_def, tf.GraphDef) else graph_def.graph_def.node
        for node in nodes_to_clear_device:
            node.device = ""
    except Exception as e:
        raise FrameworkError(
            'TensorFlow cannot read the model file: "{}" is incorrect TensorFlow model file. '
            '\nThe file should contain one of the following TensorFlow graphs:'
            '\n1. frozen graph in text or binary format'
            '\n2. inference graph for freezing with checkpoint (--input_checkpoint) in text or binary format'
            '\n3. meta graph'
            '\n\nMake sure that --input_model_is_text is provided for a model in text format. '
            'By default, a model is interpreted in binary format. Framework error details: {}. ' +
            refer_to_faq_msg(43),
            graph_file_name,
            str(e)
        ) from e
    return graph_def


def get_output_node_names_list(graph_def, user_defined_output_node_names_list: list):
    return summarize_graph(graph_def)['outputs'] \
        if user_defined_output_node_names_list is None or len(user_defined_output_node_names_list) == 0 \
        else user_defined_output_node_names_list


def deducing_metagraph_path(meta_graph_file: str):
    match = re.search('^(.*)\.(data-\d*-of-\d*|index|meta)$', meta_graph_file)
    if match is not None:
        deduced_meta_graph_file = match.group(1) + '.meta'
        if not os.path.isfile(deduced_meta_graph_file):
            raise Error('\n\nMetaGraph freezing mechanism was enabled. '
                        '\n{} file does not represent MetaGraph. '
                        '\n{} path to MetaGraph was deduced, but it does not exist'
                        '\n\nModel with MetaGraph consists of 3-4 files:'
                        '\n1. model_name.meta'
                        '\n2. model_name.index'
                        '\n3. model_name.data-00000-of-00001 (digit part may vary)'
                        '\n4. checkpoint (optional)'.format(meta_graph_file, deduced_meta_graph_file))
        else:
            meta_graph_file = deduced_meta_graph_file
    else:
        raise Error('\n\nMetaGraph freezing mechanism was enabled. '
                    '\n{} file does not represent MetaGraph. '
                    '\n\nModel with MetaGraph consists of 3-4 files:'
                    '\n1. model_name.meta'
                    '\n2. model_name.index'
                    '\n3. model_name.data-00000-of-00001 (digit part may vary)'
                    '\n4. checkpoint (optional)'
                    '\n\nTo load this model, simply run:'
                    '\npython3 mo_tf.py --input_meta_graph model_name.meta'
                    ''.format(meta_graph_file))
    return meta_graph_file


def load_tf_graph_def(graph_file_name: str = "", is_binary: bool = True, checkpoint: str = "",
                      model_dir: str = "", saved_model_tags: list = [], meta_graph_file: str = "",
                      user_output_node_names_list: list = []):
    # As a provisional solution, use a native TF methods to load a model protobuf
    graph_def = tf.GraphDef()
    if isinstance(graph_file_name, str) and (re.match('.*\.(ckpt|meta)$', graph_file_name)):
        print('[ WARNING ] The value for the --input_model command line parameter ends with ".ckpt" or ".meta" '
              'extension.\n'
              'It means that the model is not frozen.\n'
              'To load non frozen model to Model Optimizer run:'
              '\n\n1. For "*.ckpt" file:'
              '\n- if inference graph is in binary format'
              '\npython3 mo_tf.py --input_model "path/to/inference_graph.pb" --input_checkpoint "path/to/*.ckpt"'
              '\n- if inference graph is in text format'
              '\npython3 mo_tf.py --input_model "path/to/inference_graph.pbtxt" --input_model_is_text '
              '--input_checkpoint "path/to/*.ckpt"'
              '\n\n2. For "*.meta" file:'
              '\npython3 mo_tf.py --input_meta_graph "path/to/*.meta"')
    variables_values = {}
    try:
        if graph_file_name and not meta_graph_file and not checkpoint:
            # frozen graph
            return read_file_to_graph_def(graph_def, graph_file_name, is_binary), variables_values
        if graph_file_name and not meta_graph_file and checkpoint:
            # inference graph and checkpoint
            graph_def = read_file_to_graph_def(graph_def, graph_file_name, is_binary)
            outputs = get_output_node_names_list(graph_def, user_output_node_names_list)
            if os.path.isfile(checkpoint):
                graph_def = freeze_checkpoint(graph_def=graph_def, checkpoint=checkpoint, output_node_names=outputs)
            elif os.path.isdir(checkpoint):
                graph_def, variables_values = freeze_checkpoints(graph_def=graph_def, checkpoint_dir=checkpoint,
                                                                 output_node_names=outputs)
            # we are sure that checkpoint is existing file or directory due to cli_parser configuration
            return graph_def, variables_values
        if not graph_file_name and meta_graph_file:
            meta_graph_file = deducing_metagraph_path(meta_graph_file)
            input_meta_graph_def = read_file_to_graph_def(tf.MetaGraphDef(), meta_graph_file, is_binary)
            # pylint: disable=no-member
            with tf.Session() as sess:
                restorer = tf.train.import_meta_graph(input_meta_graph_def)
                restorer.restore(sess, re.sub('\.meta$', '', meta_graph_file))
                outputs = get_output_node_names_list(input_meta_graph_def.graph_def, user_output_node_names_list)
                graph_def = tf.graph_util.convert_variables_to_constants(sess, input_meta_graph_def.graph_def, outputs)
                return graph_def, variables_values
        if model_dir:
            # saved model directory
            tags = saved_model_tags if saved_model_tags is not None else [tf.saved_model.tag_constants.SERVING]
            with tf.Session() as sess:
                meta_graph_def = tf.saved_model.loader.load(sess, tags, model_dir)
                outputs = get_output_node_names_list(meta_graph_def.graph_def, user_output_node_names_list)
                graph_def = tf.graph_util.convert_variables_to_constants(sess, meta_graph_def.graph_def, outputs)
                return graph_def, variables_values
    except Exception as e:
        raise FrameworkError('Cannot load input model: {}', e) from e
    raise Error("Unknown configuration of input model parameters")


def protobuf_attrs(pb: tf.NodeDef):
    return {'pb': pb}


def protobuf2nx(pb: tf.GraphDef):
    graph = create_graph_with_nodes(pb.node, get_id=lambda pb: pb.name, get_attrs=protobuf_attrs)
    # initial order of nodes in the GraphDef. It is used to specify order in
    # which merged nodes are added to the generated sub-graph GraphDef for the TensorFlow offload feature.
    graph.graph['initial_nodes_order'] = [node.name for node in pb.node]

    # Remove data dependency edges. This is needed for the TF offload case
    for _, attrs in list(graph.nodes(data=True)):
        pb = attrs['pb']
        if '_class' in pb.attr:
            index = 0
            while index < len(pb.attr['_class'].list.s):
                if re.match('^loc:@.*', pb.attr['_class'].list.s[index].decode('utf-8')):
                    del pb.attr['_class'].list.s[index]
                else:
                    index = index + 1

    return graph


def variables_to_constants(graph: Graph, variables_values: dict):
    """
    Converts `Variable<V2>` operations to FakeConst operations with `value` from `variables_values` dictionary
    :param graph: graph to operate on
    :param variables_values: dictionary with variable names as keys and np.array data as values
    """
    for node in graph.get_op_nodes(op='FakeConst'):
        node_name = node.name

        if node_name not in variables_values:
            log.debug("There is no value for '{}': {} in checkpoint variable values".format(node.op, node_name))
            continue

        node['value'] = variables_values[node_name]
