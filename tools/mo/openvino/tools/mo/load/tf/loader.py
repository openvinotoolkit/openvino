# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

# do not print INFO and WARNING messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow.compat.v1 as tf_v1
except ImportError:
    import tensorflow as tf_v1

#in some environment suppressing through TF_CPP_MIN_LOG_LEVEL does not work
tf_v1.get_logger().setLevel("ERROR")

try:
    import tensorflow.contrib  # pylint: disable=no-name-in-module,import-error
except:
    pass  # we try to import contrib for loading models that use contrib operations

import logging as log

from openvino.tools.mo.load.loader import Loader
from openvino.tools.mo.front.common.register_custom_ops import check_for_duplicates
from openvino.tools.mo.front.common.register_custom_ops import update_extractors_with_extensions
from openvino.tools.mo.front.extractor import restore_edges, extract_node_attrs, remove_control_dependency_inputs, add_outputs_identity
from openvino.tools.mo.front.tf.extractor import get_tf_edges, create_tf_edge, tf_op_extractor, tf_op_extractors
from openvino.tools.mo.front.tf.loader import load_tf_graph_def, protobuf2nx
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from openvino.tools.mo.utils import tensorboard_util
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.telemetry_utils import send_op_names_info, send_shapes_info, send_framework_info
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class TFLoader(Loader):
    enabled = True
    run_not_recursively = True

    def load(self, graph: Graph):
        argv = graph.graph['cmd_params']
        if argv.tensorflow_custom_layer_libraries:
            libraries = argv.tensorflow_custom_layer_libraries.split(',')
            for library in libraries:
                log.info('Loading library "{}" with custom operations'.format(library))
                tf_v1.load_op_library(library)

        graph_def, variables_values, framework, inputs_outputs_order = load_tf_graph_def(
            graph_file_name=argv.input_model,
            is_binary=not argv.input_model_is_text,
            checkpoint=argv.input_checkpoint,
            user_output_node_names_list=argv.output,
            model_dir=argv.saved_model_dir,
            meta_graph_file=argv.input_meta_graph,
            saved_model_tags=argv.saved_model_tags)

        if inputs_outputs_order is not None and isinstance(inputs_outputs_order, tuple):
            graph.inputs_order = inputs_outputs_order[0]
            graph.outputs_order = inputs_outputs_order[1]

        send_framework_info(framework)

        try:
            tf_v1.import_graph_def(graph_def, name='')
        except:
            log.warning("TensorFlow post-processing of loaded model was unsuccessful. "
                        "This is an optional step that Model Optimizer performs for any input model but it is not usually "
                        "required for all models. "
                        "It likely means that the original model is ill-formed. "
                        "Model Optimizer will continue converting this model.")

        log.debug("Number of nodes in graph_def: {}".format(len(graph_def.node)))  # pylint: disable=no-member

        if argv.tensorboard_logdir:
            tensorboard_util.dump_for_tensorboard(graph_def, argv.tensorboard_logdir)

        update_extractors_with_extensions(tf_op_extractors)

        try:
            protobuf2nx(graph, graph_def)
        except Exception as e:
            raise Error(
                'Cannot pre-process TensorFlow graph after reading from model file "{}". ' \
                'File is corrupt or has unsupported format. Details: {}. ' +
                refer_to_faq_msg(44),
                argv.model_name,
                str(e)
            ) from e

        graph.__setattr__('name', argv.model_name)
        # 'layout' parameter change may cause an issue in EltwiseInputReshape replacer
        # and convert_nhwc_to_nchw(graph)
        graph.graph['layout'] = 'NHWC'
        graph.graph['fw'] = 'tf'

        graph.graph['variables_values'] = variables_values
        del variables_values

        used_tensors = restore_edges(graph, get_tf_edges)

        # Tensor names information corresponding to a node is stored on outgoing edges.
        # As output nodes do not have outgoing edges, fake outputs are required. In the following code
        # for each output Identity node is added, and tensor name for the output is kept
        # on (output, fake output) edge. After Result nodes adding transformation fake outputs
        # are deleted from graph.
        add_outputs_identity(graph, graph.nodes - used_tensors, lambda g, output, fake_node_name: g.add_edges_from([
            create_tf_edge(output, fake_node_name, 0)]))

        remove_control_dependency_inputs(graph)

        graph.check_empty_graph('protobuf2nx. It may happen due to problems with loaded model')
        extract_node_attrs(graph, lambda node: tf_op_extractor(node, check_for_duplicates(tf_op_extractors)))

        # try to detect layout from the nodes of the graph. If there are no convolution nodes in N(D)HWC layout then we
        # consider that the graph is in NCHW layout and no layout conversion should be performed
        if not graph_or_sub_graph_has_nhwc_ops(graph):
            if not argv.silent:
                log.debug('disable_nhwc_to_nchw" was automatically enabled.')
            for_graph_and_each_sub_graph_recursively(graph, update_cmd_params_and_layout)

        send_op_names_info(framework, graph)
        send_shapes_info(framework, graph)


def is_node_layout_nhwc(node: Node):
    """
    Check the layout attribute of specific operations and return True if any of them has layout NHWC.
    :param node: Node to check
    :return: Boolean result of the check
    """
    if node.soft_get('op') in ["Conv2D", "DepthwiseConv2dNative", "Conv3D", "Conv2DBackpropInput",
                               "Conv3DBackpropInputV2"]:
        if node.soft_get('layout') in ["NHWC", "NDHWC"]:
            log.debug('Detected convolution node with NHWC layout: "{}"'.format(node.soft_get('name', node.id)))
            return True
    return False


def graph_or_sub_graph_has_nhwc_ops(graph: Graph):
    """
    Checks that a graph or any sub-graph (inside Loop) operation contains nodes with NHWC layout.
    :param graph: main graph to check
    :return: Boolean result of the check
    """
    NHWC_conv_detected = False
    for node in graph.get_op_nodes():
        if is_node_layout_nhwc(node):
            NHWC_conv_detected = True
            break

        if node.has('sub_graphs'):
            for sub_graph_name in node['sub_graphs']:
                NHWC_conv_detected |= graph_or_sub_graph_has_nhwc_ops(node.soft_get(sub_graph_name))

    return NHWC_conv_detected


def update_cmd_params_and_layout(graph: Graph):
    """
    Updates "cmd_params" and "layout" attribute as the model has only NCHW layout operations.
    :param graph: graph to update attributes
    :return: Nones
    """
    if 'cmd_params' in graph.graph:
        graph.graph['cmd_params'].auto_disable_nhwc_to_nchw = True
    if 'layout' in graph.graph:
        graph.graph['layout'] = 'NCHW'
