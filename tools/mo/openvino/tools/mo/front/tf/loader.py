# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import os
import re
from distutils.version import LooseVersion

from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error, FrameworkError
from openvino.tools.mo.utils.utils import refer_to_faq_msg
from openvino.tools.mo.utils.versions_checker import get_environment_setup

# do not print INFO and WARNING messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow.compat.v1 as tf_v1
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
except ImportError:
    import tensorflow as tf_v1

# in some environment suppressing through TF_CPP_MIN_LOG_LEVEL does not work
tf_v1.get_logger().setLevel("ERROR")

from google.protobuf import text_format
from openvino.tools.mo.graph.graph import fill_graph_with_nodes, Graph
from openvino.tools.mo.utils.summarize_graph import summarize_graph


def freeze_checkpoints(graph_def: tf_v1.GraphDef, checkpoint_dir: str, output_node_names: list):
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

    tf_v1.import_graph_def(graph_def, name='')

    with tf_v1.Session() as sess:
        uninitialized_variables = [str(v, 'utf-8') for v in set(sess.run(tf_v1.report_uninitialized_variables()))]
        all_variables = [n.name for n in sess.graph.as_graph_def().node if n.op in ['Variable', 'VariableV2']]
        white_list = [v for v in all_variables if v not in uninitialized_variables]
        black_list = [v for v in all_variables if v in uninitialized_variables]
        output_graph_def = tf_v1.graph_util.convert_variables_to_constants(sess, graph_def, output_node_names,
                                                                           variable_names_whitelist=white_list,
                                                                           variable_names_blacklist=black_list)
    variable_values = {}
    for checkpoint_file in checkpoint_files:
        log.debug("Loading {}".format(checkpoint_file))
        with tf_v1.Session() as sess:
            var_list = {}
            var_to_shape_map = tf_v1.train.load_checkpoint(checkpoint_file).get_variable_to_shape_map()
            for key in var_to_shape_map:
                try:
                    tensor = sess.graph.get_operation_by_name(key).outputs[0]
                except KeyError:
                    continue
                var_list[key] = tensor
            tf_v1.train.Saver(var_list=var_list).restore(sess, checkpoint_file)
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
    tf_v1.import_graph_def(graph_def, name="")

    with tf_v1.Session() as sess:
        var_list = {}
        var_to_shape_map = tf_v1.train.NewCheckpointReader(checkpoint).get_variable_to_shape_map()
        for key in var_to_shape_map:
            try:
                tensor = sess.graph.get_operation_by_name(key).outputs[0]
            except KeyError:
                continue
            var_list[key] = tensor
        tf_v1.train.Saver(var_list=var_list).restore(sess, checkpoint)
        output_graph_def = tf_v1.graph_util.convert_variables_to_constants(sess, graph_def, output_node_names)
    return output_graph_def


def read_file_to_graph_def(graph_def: [tf_v1.GraphDef, tf_v1.MetaGraphDef], graph_file_name: str = "",
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
        nodes_to_clear_device = graph_def.node if isinstance(graph_def, tf_v1.GraphDef) else graph_def.graph_def.node
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
    match = re.search(r'^(.*)\.(data-\d*-of-\d*|index|meta)$', meta_graph_file)
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


def freeze_tf2_concrete_function(model, concrete_func, env_setup):

    if "tensorflow" in env_setup and env_setup["tensorflow"] >= LooseVersion("2.2.0"):
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=False,
                                                        aggressive_inlining=True)  # pylint: disable=E1123
    else:
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=False)  # pylint: disable=E1123
    graph_def = frozen_func.graph.as_graph_def(add_shapes=True)

    input_names = []
    if hasattr(model, 'inputs') and model.inputs is not None:
        # Extract tensor names order from Keras model
        input_names = [tensor.name for tensor in model.inputs]

    # After model freezing output tensor names are changing and recieve "Func/PartitionedCall" prefix,
    # so output_names from saved_model cannot be used. Here tensor names from frozen graph are used,
    # as TF adds indexed Identity nodes during freezing to each output, so this indexing is used for
    # order alignment.
    output_names = [tensor.name for tensor in frozen_func.outputs]

    inputs_outputs_order = (input_names, output_names)

    return graph_def, {}, 'tf2', inputs_outputs_order


def prepare_graph_def(model):
    from tensorflow.python.training.tracking.base import Trackable  # pylint: disable=no-name-in-module,import-error
    if isinstance(model, tf_v1.GraphDef):
        nodes_to_clear_device = model.node
        for node in nodes_to_clear_device:
            node.device = ""
        return model, {}, "tf", None
    if isinstance(model, tf.keras.Model):
        env_setup = get_environment_setup("tf")

        assert hasattr(model, "inputs") and model.inputs is not None, "Model inputs specification is required."

        model_inputs = []
        for inp in model.inputs:
            if isinstance(inp, tf.Tensor):
                model_inputs.append(inp)
            elif tf.keras.backend.is_keras_tensor(inp):
                model_inputs.append(inp.type_spec)
            else:
                raise Error("Unknown input tensor type {}".format(type(input)))

        @tf.function
        def tf_function(x):
            return model(x)

        conc_func = tf_function.get_concrete_function(model_inputs)
        return freeze_tf2_concrete_function(model, conc_func, env_setup)
    if isinstance(model, Trackable):
        env_setup = get_environment_setup("tf")
        return saved_model_load(model, env_setup)
    raise Exception("Unknown model type {}.".format(type(model)))


def saved_model_load(imported, env_setup):
    # to get a signature by key throws KeyError for TF 1.x SavedModel format in case TF 2.x installed
    concrete_func = imported.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    # the aggressive inlining parameter needs to freeze a table of embeddings for Keras Embedding operation
    # and a model with Embedding operation cannot properly converted to IR without this function parameter

    return freeze_tf2_concrete_function(imported, concrete_func, env_setup)


def load_tf_graph_def(graph_file_name: str = "", is_binary: bool = True, checkpoint: str = "",
                      model_dir: str = "", saved_model_tags: list = [], meta_graph_file: str = "",
                      user_output_node_names_list: list = []):

    if not isinstance(graph_file_name, str) and graph_file_name is not None:
        return prepare_graph_def(graph_file_name)
    # As a provisional solution, use a native TF methods to load a model protobuf
    graph_def = tf_v1.GraphDef()
    if isinstance(graph_file_name, str) and (re.match(r'.*\.(ckpt|meta)$', graph_file_name)):
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
            return read_file_to_graph_def(graph_def, graph_file_name, is_binary), variables_values, 'tf', None
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
            return graph_def, variables_values, 'tf', None
        if not graph_file_name and meta_graph_file:
            meta_graph_file = deducing_metagraph_path(meta_graph_file)
            input_meta_graph_def = read_file_to_graph_def(tf_v1.MetaGraphDef(), meta_graph_file, is_binary)
            # Since version 2.2 TF can fail with internal error while loading graph from .meta file.
            # It happens because some operation may has an _output_shapes attribute inconsistent with the GraphDef
            # calculated value. To avoid this problem we must delete `_output_shapes` attributes from operations
            for node in input_meta_graph_def.graph_def.node:
                if '_output_shapes' in node.attr:
                    del node.attr['_output_shapes']
            # pylint: disable=no-member
            with tf_v1.Session() as sess:
                restorer = tf_v1.train.import_meta_graph(input_meta_graph_def)
                restorer.restore(sess, re.sub(r'\.meta$', '', meta_graph_file))
                outputs = get_output_node_names_list(input_meta_graph_def.graph_def, user_output_node_names_list)
                graph_def = tf_v1.graph_util.convert_variables_to_constants(sess, input_meta_graph_def.graph_def,
                                                                            outputs)
                return graph_def, variables_values, 'tf', None
        if model_dir:
            # saved model directory
            try:
                env_setup = get_environment_setup("tf")

                try:
                    # Code to extract Keras model.
                    # tf.keras.models.load_model function throws TypeError,KeyError or IndexError
                    # for TF 1.x SavedModel format in case TF 1.x installed
                    imported = tf.keras.models.load_model(model_dir, compile=False)
                except:
                    imported = tf.saved_model.load(model_dir, saved_model_tags)  # pylint: disable=E1120

                return saved_model_load(imported, env_setup)
            except:
                # code to extract GraphDef for TF 1.0 SavedModel format
                tags = saved_model_tags if saved_model_tags is not None else [tf_v1.saved_model.tag_constants.SERVING]
                with tf_v1.Session() as sess:
                    meta_graph_def = tf_v1.saved_model.loader.load(sess, tags, model_dir)
                    outputs = get_output_node_names_list(meta_graph_def.graph_def, user_output_node_names_list)
                    graph_def = tf_v1.graph_util.convert_variables_to_constants(sess, meta_graph_def.graph_def, outputs)
                    return graph_def, variables_values, 'tf', None
    except Exception as e:
        raise FrameworkError('Cannot load input model: {}', e) from e
    raise Error("Unknown configuration of input model parameters")


def convert_to_pb(argv: argparse.Namespace):
    from openvino.tools.mo.utils.cli_parser import get_model_name

    # if this is already binary frozen format .pb, there is no need to create auxiliary binary frozen protobuf
    # the main thing is to differentiate this format from text frozen format and checkpoint
    # that can utilize input_model option
    if argv.input_model and not argv.input_model_is_text and not argv.input_checkpoint and \
            isinstance(argv.input_model, str):
        return None

    user_output_node_names_list = argv.output if argv.output else None
    if user_output_node_names_list is not None and not isinstance(user_output_node_names_list, list):
        user_output_node_names_list = user_output_node_names_list.split(',')
    graph_def, _, _, _ = load_tf_graph_def(
        graph_file_name=argv.input_model,
        is_binary=not argv.input_model_is_text,
        checkpoint=argv.input_checkpoint,
        user_output_node_names_list=user_output_node_names_list,
        model_dir=argv.saved_model_dir,
        meta_graph_file=argv.input_meta_graph,
        saved_model_tags=argv.saved_model_tags)
    if argv.model_name:
        model_name = argv.model_name
    elif argv.input_model:
        model_name = get_model_name(argv.input_model)
    elif argv.saved_model_dir:
        model_name = "saved_model"
    elif argv.input_meta_graph:
        model_name = get_model_name(argv.input_meta_graph)
    argv.model_name = model_name
    tf_v1.io.write_graph(graph_def, argv.output_dir if argv.output_dir != '.' else os.getcwd(),
                         model_name + "_tmp.pb", as_text=False)
    path_to_pb = os.path.normpath(os.path.join(argv.output_dir, model_name + "_tmp.pb"))
    argv.input_model = path_to_pb
    return path_to_pb


def protobuf_attrs(pb: tf_v1.NodeDef):
    return {'pb': pb}


def protobuf2nx(graph, pb: tf_v1.GraphDef):
    fill_graph_with_nodes(graph, pb.node, get_id=lambda pb: pb.name, get_attrs=protobuf_attrs)

    if hasattr(graph, 'op_names_statistic'):
        for node_name in graph.nodes:
            node = Node(graph, node_name)
            node_pb = node.soft_get('pb', None)
            if node_pb is not None:
                if hasattr(node_pb, 'op'):
                    graph.op_names_statistic[node_pb.op] += 1

    # Create a library with auxiliary functions used in TensorFlow 2 operations
    if hasattr(pb, 'library') and hasattr(pb.library, 'function'):
        graph.graph['library'] = {}
        for library_function in pb.library.function:
            function_name = library_function.signature.name
            graph.graph['library'][function_name] = {}
            graph.graph['library'][function_name]['input_arg'] = library_function.signature.input_arg
            graph.graph['library'][function_name]['output_arg'] = library_function.signature.output_arg
            graph.graph['library'][function_name]['node_def'] = library_function.node_def
            graph.graph['library'][function_name]['ret'] = library_function.ret
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
