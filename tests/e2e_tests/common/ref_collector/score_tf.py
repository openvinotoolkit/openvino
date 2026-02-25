# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from e2e_tests.common.ref_collector.provider import ClassProvider
from e2e_tests.utils.path_utils import resolve_file_path, resolve_dir_path
import os
import re
import sys
import itertools
from collections import defaultdict
import logging as log

os.environ['GLOG_minloglevel'] = '3'


def build_control_flow_children_map(graph):
    """
    Builds map: graph.node -> set of nodes that have incoming control flow
    dependencies from graph.node.
    """
    mapping = defaultdict(set)
    ops = graph.get_operations()
    for op in ops:
        for src in op.control_inputs:
            mapping[src].add(op)
    return mapping


def trace_loop(enter, control_flow_map):
    """
    Starting from enter traverse graph nodes until face Exit op, if Enter is
    discovered, do trace_loop for it. Returns all discovered tensors inside the
    loop(s).
    """
    for_examination = set(enter.outputs[0].consumers())  # ops
    visited = set()
    collected = set(enter.outputs)  # tensors
    exits = set()  # ops
    while len(for_examination):
        candidate = for_examination.pop()
        if candidate in visited:
            continue
        visited.add(candidate)
        if candidate.type == 'Exit':
            exits.add(candidate)
            continue
        if candidate.type == 'Enter':
            # nested loop is detected
            nested_collected, nested_exits = trace_loop(candidate,
                                                        control_flow_map)
            for_examination = for_examination | nested_exits
            collected = collected | nested_collected
        else:
            collected = collected | set(candidate.outputs)
            for_examination = for_examination | set(
                itertools.chain.from_iterable(
                    [output.consumers() for output in candidate.outputs]))
            for_examination = for_examination | control_flow_map[candidate]
    return collected, exits


def find_all_tensors_in_loops(graph):
    """Search for all Enter operations in the graph."""
    ops = graph.get_operations()
    enters = [op for op in ops if op.type == 'Enter']
    collected = set()
    control_flow_map = build_control_flow_children_map(graph)
    for enter in enters:
        nested_collected, _ = trace_loop(enter, control_flow_map)
        collected = collected | nested_collected
    return collected


def children(op_name: str, graph):
    """Get operation node children."""
    op = graph.get_operation_by_name(op_name)
    return set(op for out in op.outputs for op in out.consumers())


def summarize_graph(graph_def):
    import tensorflow as tf
    unlikely_output_types = ['Const', 'Assign', 'NoOp', 'Placeholder', 'Assert', 'switch_t', 'switch_f']
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


def get_output_node_names_list(graph_def, user_defined_output_node_names_list: list):
    return summarize_graph(graph_def)['outputs'] if len(user_defined_output_node_names_list) == 0 \
        else user_defined_output_node_names_list


class ScoreTensorFlowBase(ClassProvider):
    """Reference collector for TensorFlow models."""
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.output_nodes_for_freeze = config.get("output_nodes_for_freeze", None)
        self.additional_outputs = config.get("additional_outputs", [])
        self.override_default_outputs = config.get("override_default_outputs", False)
        self.additional_inputs = config.get("additional_inputs", [])
        self.user_output_node_names_list = config.get("user_output_node_names_list", [])
        self.override_default_inputs = config.get("override_default_inputs", False)
        self.inputs = config["inputs"]
        self.res = {}

    def load_graph(self):
        """
        load_graph function have to be implemented in inherited classes
        depending on type of input tf model (from saved_dir, from meta or simple pb)
        """
        raise NotImplementedError("{}\nDo not use {} class directly!".format(self.load_graph().__doc__,
                                                                             self.__class__.__name__))

    def get_refs(self):
        """Return TensorFlow model reference results."""
        log.info("Running inference with tensorflow ...")
        import tensorflow as tf
        graph = self.load_graph()
        feed_dict = {}
        summary_info = summarize_graph(graph.as_graph_def())

        input_layers, output_layers = list(summary_info['inputs'].keys()), summary_info['outputs']
        if self.override_default_outputs and self.additional_outputs:
            output_layers = self.additional_outputs
        else:
            output_layers.extend(self.additional_outputs)
        if self.override_default_inputs and self.additional_inputs:
            input_layers = self.additional_inputs
        else:
            input_layers.extend(self.additional_inputs)
        data_keys = [key for key in self.inputs.keys()]
        if sorted(input_layers) != sorted(data_keys):
            raise ValueError('input data keys: {data_keys} do not match input '
                             'layers of network: {input_layers}'.format(data_keys=data_keys, input_layers=input_layers))

        for input_layer_name in input_layers:
            # Case when port is already in layer name
            port = re.search(r':[0-9]*$', input_layer_name)
            if port is not None:
                tensor = graph.get_tensor_by_name(input_layer_name)
            else:
                tensor = graph.get_tensor_by_name(input_layer_name + ':0')
            feed_dict[tensor] = self.inputs[input_layer_name]
        output_tensors = []
        for name in output_layers:
            tensor = graph.get_tensor_by_name(name + ':0')
            output_tensors.append(tensor)

        log.info("Running tf.Session")
        with graph.as_default():
            with tf.compat.v1.Session(graph=graph) as session:
                outputs = session.run(output_tensors, feed_dict=feed_dict)
        self.res = dict(zip(output_layers, outputs))
        log.info("TensorFlow reference collected successfully\n")
        return self.res


class ScoreTensorFlow(ScoreTensorFlowBase):
    __action_name__ = "score_tf"

    def __init__(self, config):
        self.model = resolve_file_path(config["model"], as_str=True)
        super().__init__(config=config)

    def load_graph(self):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()

        with open(self.model, "rb") as model_file:
            graph_def.ParseFromString(model_file.read())

        nodes_to_clear_device = graph_def.node if isinstance(
            graph_def, tf.compat.v1.GraphDef) else graph_def.graph_def.node
        for node in nodes_to_clear_device:
            node.device = ""

        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

        log.info("tf graph was created")
        return graph


class ScoreTensorFlowMeta(ScoreTensorFlowBase):
    __action_name__ = "score_tf_meta"

    def __init__(self, config):
        self.model = resolve_file_path(config["model"], as_str=True)
        super().__init__(config=config)

    def load_graph(self):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        graph_def = tf.compat.v1.MetaGraphDef()

        with open(self.model, "rb") as model_file:
            graph_def.ParseFromString(model_file.read())

        nodes_to_clear_device = graph_def.node if isinstance(
            graph_def, tf.compat.v1.GraphDef) else graph_def.graph_def.node
        for node in nodes_to_clear_device:
            node.device = ""

        assert bool(self.output_nodes_for_freeze), \
            "Input model has .meta extension. To freeze model need to specify 'output_nodes_for_freeze'"
        log.info("Created tf.Session")
        with tf.compat.v1.Session() as sess:
            restorer = tf.compat.v1.train.import_meta_graph(graph_def)
            restorer.restore(sess, re.sub('\.meta$', '', self.model))
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, graph_def.graph_def, self.output_nodes_for_freeze)

        with graph.as_default():
            tf.import_graph_def(graph_def, name='')

        log.info("tf graph was created")
        return graph


class ScoreTensorFlowFromDir(ScoreTensorFlowBase):
    __action_name__ = "score_tf_dir"

    def __init__(self, config):
        self.model = resolve_dir_path(config["model"], as_str=True)
        super().__init__(config=config)

    def load_graph(self):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        tags = [tf.saved_model.SERVING]
        log.info("Created tf.Session")
        with tf.compat.v1.Session() as sess:
            meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, tags, self.model)
            outputs = get_output_node_names_list(meta_graph_def.graph_def, self.user_output_node_names_list)
            graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, meta_graph_def.graph_def, outputs)
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(graph_def, name='')
        log.info("tf graph was created")
        return graph