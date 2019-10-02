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

import json
import logging as log
import os
from re import compile, match

import networkx as nx

from mo.graph.graph import Node, Graph
from mo.utils.error import Error
from mo.utils.graph import nodes_matching_name_pattern, sub_graph_between_nodes
from mo.utils.utils import refer_to_faq_msg


class CustomReplacementDescriptor(object):
    registered_types = dict()

    def __init__(self, replacement_id: str, attrs: dict = None):
        """
        Create class instance based on attrs dictionary which is read from the configuration file.
        :param attrs:
        """
        super(CustomReplacementDescriptor, self).__setattr__('replacement_id', replacement_id)
        if attrs is not None:
            super(CustomReplacementDescriptor, self).__setattr__('custom_attributes',
                                                                 attrs.setdefault('custom_attributes', {}))
            super(CustomReplacementDescriptor, self).__setattr__('_replacement_desc', attrs.copy())

    def __getattr__(self, k):
        return self._replacement_desc[k]

    def __setattr__(self, k, v):
        # you can assign only existing attributes
        if k not in self._replacement_desc:
            raise AttributeError
        self._replacement_desc[k] = v

    def has(self, attr):
        """
        Check that attribute 'attr' is defined for the CustomReplacementDescriptor.
        :param attr: attribute to check.
        :return: True if the attribute exists and False otherwise.
        """
        return attr in self._replacement_desc

    @classmethod
    def register_type(cls, match_kind: str, class_type: object):
        if match_kind in cls.registered_types:
            log.warning('Class for match kind "{}" is already registered'.format(match_kind))
        else:
            cls.registered_types[match_kind] = class_type

    @classmethod
    def create_instance(cls, match_kind: str, replacement_id: str, attrs: dict = None):
        """
        Fabric method to create proper object based on match_kind.
        :param match_kind: match kind.
        :param replacement_id: id of the replacement.
        :param attrs: optional attributes to be set.
        :return: object of the sub-class of the CustomLayerDescriptor class or None if the match kind is not registered.
        """
        if attrs is None:
            attrs = dict()
        if match_kind in cls.registered_types:
            return cls.registered_types[match_kind](replacement_id, attrs)
        else:
            raise Error('No class registered for match kind "{}". Supported match kinds are "{}". '.format(
                match_kind, list(cls.registered_types.keys())) +
                        refer_to_faq_msg(65))

    def sub_graph_instances(self):
        raise Exception("The function 'get_sub_graph_instances' must be implemented in the sub-class.")

    def get_config_file_representation(self):
        result = {
            'match_kind': self.match_kind, 'instances': self.instances,
            'inputs': self.inputs, 'outputs': self.outputs,
            'custom_attributes': self.custom_attributes, 'id': self.id
        }
        if self.has('op'):
            result.update({'op': self.op})
        return result

    def get_inputs_description(self):
        """
        Returns description of inputs of the layer with id 'layer_id'. The format of inputs is the following: list of
        lists where each list contains information about nodes consuming the same tensor from outside of the graph. Each
        element of the list is a pair where first element is a regular expression for the name of the node in the
        sub-graph and the second is the input port of this node.
        :return: description of inputs or None if layer with such id is not registered or information about inputs is
        not available.
        """
        if 'inputs' not in self._replacement_desc:
            log.error("Information about inputs of layer with id '{}' is not available".format(self.replacement_id))
            return None
        result = list()
        for index, input_desc in enumerate(self._replacement_desc['inputs']):
            result.append([(inp['node'], inp['port']) for inp in input_desc])
        return result

    def get_outputs_description(self):
        """
        Returns description of outputs of the layer with id 'layer_id'. The format of outputs is the following: list of
        pairs where the first element of the pair is a regular expression for the name of the node that produces output
        of the sub-graph and the second is the output port of this node.
        :return: description of outputs or None if layer with such id is not registered or information about outputs is
        not available.
        """
        if 'outputs' not in self._replacement_desc:
            log.error("Information about outputs of layer with id '{}' is not available")
            return None
        return [(out['node'], out['port']) for out in self._replacement_desc['outputs']]

    def update_custom_replacement_attributes(self, graph: Graph):
        """
        The function run specific functions to update attributes of the custom replacement description. Currently it
        updates information about input/output nodes.
        :param graph: graph to operate on.
        :return: True if the update process completed successfully.
        """
        raise Exception("The function 'update_custom_layer_attributes' must be implemented in the sub-class.")

    def validate_data(self):
        """
        Validates layer description dictionary.
        :return: list of errors identified.
        """
        errors = list()
        if not self.has('id'):
            errors.append("Replacement id is not specified for custom replacement '{}'".format(self.replacement_id))
        if not self.has('instances') or self.instances == '':
            errors.append("Attribute 'instances' is not specified for replacement '{}'".format(self.replacement_id))
        if not self.has('match_kind'):
            errors.append("Replacement match type is not specified for replacement '{}'".format(self.replacement_id))
        return errors


class CustomReplacementDescriptorPoints(CustomReplacementDescriptor):
    """
    Class that is used to describe custom replacement which is a sub-graph specified by start and end points.
    """

    def __init__(self, replacement_id: str, attrs: dict = None):
        super().__init__(replacement_id, attrs)
        if not self.has('include_inputs_to_sub_graph'):
            super(CustomReplacementDescriptorPoints, self).__setattr__('include_inputs_to_sub_graph', True)
        if not self.has('include_outputs_to_sub_graph'):
            super(CustomReplacementDescriptorPoints, self).__setattr__('include_outputs_to_sub_graph', True)

    def get_config_file_representation(self):
        result = {
            'match_kind': self.match_kind, 'instances': self.instances,
            'custom_attributes': self.custom_attributes, 'id': self.id,
            'include_inputs_to_sub_graph': bool(self.include_inputs_to_sub_graph),
            'include_outputs_to_sub_graph': bool(self.include_outputs_to_sub_graph)
        }
        if self.has('op'):
            result.update({'op': self.op})
        return result

    def get_inputs_description(self):
        return [[('^' + node_name + '$', 0)] for node_name in self.instances['start_points']]

    def get_outputs_description(self):
        return [('^' + node_name + '$', 0) for node_name in self.instances['end_points']]

    def get_internal_input_nodes(self, graph: Graph):
        """
        Gets list of node names getting input from outside of the sub-graph. This function checks whether input nodes
        specified in the configuration file should be added to the sub-graph or not. If they should not be added to the
        sub-graph then input nodes of the sub-graph are children of these nodes.
        :param graph: graph to operate on.
        :return: list of input node names.
        """
        if not self.include_inputs_to_sub_graph:
            log.debug('Do not include inputs to sub-graph for replacement with id {}'.format(self.replacement_id))
            new_start_nodes = set()
            for start_node in self.instances['start_points']:
                for _, out_node_name in graph.out_edges(start_node):
                    new_start_nodes.add(out_node_name)
            start_nodes = list(new_start_nodes)
            log.debug('New inputs are: {}'.format(start_nodes))
            return start_nodes
        else:
            return self.instances['start_points']

    def get_internal_output_nodes(self, graph: Graph):
        """
        Gets list of node names producing output outside of the sub-graph. This function checks whether output nodes
        specified in the configuration file should be added to the sub-graph or not. If they should not be added to the
        sub-graph then output nodes of the sub-graph are parents of these nodes.
        :param graph: graph to operate on.
        :return: list of output node names.
        """
        if not self.include_outputs_to_sub_graph:
            log.debug('Do not include outputs of sub-graph for replacement with id {}'.format(self.replacement_id))
            new_end_nodes = set()
            for end_node in self.instances['end_points']:
                for in_node_name, _ in graph.in_edges(end_node):
                    new_end_nodes.add(in_node_name)
            end_nodes = list(new_end_nodes)
            log.debug('New outputs are: {}'.format(end_nodes))
            return end_nodes
        else:
            return self.instances['end_points']

    def update_custom_replacement_attributes(self, graph: Graph):
        if not self.has('instances'):
            raise Error("No instance(s) is(are) defined for the custom replacement '{}'. ".format(self.replacement_id) +
                        refer_to_faq_msg(66))
        if not isinstance(self.instances, dict):
            raise Error("The instance must be a single dictionary for the custom replacement with id '{}'. ".format(
                self.replacement_id) +
                        refer_to_faq_msg(67))

        start_points = self.get_internal_input_nodes(graph)
        end_points = self.get_internal_output_nodes(graph)

        matched_nodes = sub_graph_between_nodes(graph, start_points, end_points)
        output_tensors = set()
        input_nodes_mapping = dict()  # key is the input tensor name, value is the pair: (input_port, output_node_name)
        for src_node_name, dst_node_name, edge_attrs in graph.edges(data=True):
            dst_node = graph.node[dst_node_name]

            # edge outside sub-graph into sub-graph
            if (src_node_name not in matched_nodes) and (dst_node_name in matched_nodes):
                tensor_name = src_node_name + ":" + str(edge_attrs['out'])
                if tensor_name not in input_nodes_mapping:
                    input_nodes_mapping[tensor_name] = list()
                input_nodes_mapping[tensor_name].append(('^' + dst_node_name + '$', edge_attrs['in']))

            # edge from inside sub-graph to outside sub-graph
            if (src_node_name in matched_nodes) and (dst_node_name not in matched_nodes):
                output_tensors.add(('^' + dst_node['pb'].input[edge_attrs['in']] + '$', edge_attrs['out']))

        for node_name in graph.nodes():
            node = Node(graph, node_name)
            if node_name in matched_nodes and len(node.out_nodes()) == 0 and node['pb'].op != 'Const':
                log.debug("Node {} doesn't have output edges. Consider it output".format(node_name))
                output_tensors.add(('^' + node_name + '$', 0))

        if not self.has('inputs'):
            self._replacement_desc['inputs'] = [[{'node': desc[0], 'port': desc[1]} for desc in inp]
                                                for inp in sorted(input_nodes_mapping.values())]
            log.debug('Updated inputs of sub-graph for instance "{}"'.format(self.instances))

        if not self.has('outputs'):
            self._replacement_desc['outputs'] = [{'node': node, 'port': port} for node, port in sorted(output_tensors)]
            log.debug('Updated outputs of sub-graph for instance "{}"'.format(self.instances))

    def sub_graph_instances(self):
        return [self.instances]


CustomReplacementDescriptor.register_type('points', CustomReplacementDescriptorPoints)


class CustomReplacementDescriptorScope(CustomReplacementDescriptor):
    """
    Class that is used to describe custom layer which is a sub-graph specified by scope name.
    """

    def __init__(self, replacement_id: str, attrs: dict = None):
        super().__init__(replacement_id, attrs)

    def update_custom_replacement_attributes(self, graph: Graph):
        if not self.has('instances') or len(self.instances) == 0:
            raise Error("No instances are defined for replacement with id '{}'. ".format(self.replacement_id) +
                        refer_to_faq_msg(68))

        pattern = self.instances[0]  # use the first instance pattern to find input/output nodes patterns
        # TODO verify that all instances will produce the same sub-graph
        matched_nodes = nodes_matching_name_pattern(graph, pattern)

        output_tensors = set()
        input_nodes_mapping = dict()  # key is the input tensor name, value is the pair: (input_port, output_node_name)
        for src_node_name, dst_node_name, edge_attrs in graph.edges(data=True):
            dst_node = graph.node[dst_node_name]

            # edge outside sub-graph into sub-graph
            if (src_node_name not in matched_nodes) and (dst_node_name in matched_nodes):
                tensor_name = src_node_name + ":" + str(edge_attrs['out'])
                if tensor_name not in input_nodes_mapping:
                    input_nodes_mapping[tensor_name] = list()
                input_nodes_mapping[tensor_name].append((generate_pattern_for_node(graph, pattern, dst_node_name),
                                                         edge_attrs['in']))

            # edge from inside sub-graph to outside sub-graph
            if (src_node_name in matched_nodes) and (dst_node_name not in matched_nodes):
                output_tensors.add(
                    (generate_pattern_for_node(graph, pattern, dst_node['pb'].input[edge_attrs['in']]),
                     edge_attrs['out']))

        for node_name in graph.nodes():
            node = Node(graph, node_name)
            if node_name in matched_nodes and len(node.out_nodes()) == 0 and node['pb'].op != 'Const':
                log.debug("Node {} doesn't have output edges. Consider it output".format(node_name))
                output_tensors.add((generate_pattern_for_node(graph, pattern, node_name), 0))

        if not self.has('inputs'):
            self._replacement_desc['inputs'] = [[{'node': desc[0], 'port': desc[1]} for desc in inp]
                                                for inp in sorted(input_nodes_mapping.values())]
            log.debug('Updated inputs of sub-graph for instance "{}"'.format(self.instances))

        if not self.has('outputs'):
            self._replacement_desc['outputs'] = [{'node': node, 'port': port} for node, port in sorted(output_tensors)]
            log.debug('Updated outputs of sub-graph for instance "{}"'.format(self.instances))

    def sub_graph_instances(self):
        return self.instances


CustomReplacementDescriptor.register_type('scope', CustomReplacementDescriptorScope)


class CustomReplacementDescriptorGeneral(CustomReplacementDescriptor):
    def __init__(self, replacement_id: str, attrs: dict = None):
        super().__init__(replacement_id, attrs)

    def validate_data(self):
        """
        Validates layer description dictionary.
        :return: list of errors identified.
        """
        errors = list()
        if not self.has('id'):
            errors.append("Replacement id is not specified for custom replacement '{}'".format(self.replacement_id))
        if not self.has('match_kind'):
            errors.append("Replacement match type is not specified for replacement '{}'".format(self.replacement_id))
        return errors


CustomReplacementDescriptor.register_type('general', CustomReplacementDescriptorGeneral)


def parse_custom_replacement_config_file(file_name: str):
    """
    Reads custom replacement configuration file file_name.
    :param file_name: name of the file to read from.
    :return: The dictionary where key is the layer id and value is an instance of the CustomLayerDescriptor object.
    """
    if not os.path.exists(file_name):
        raise Error("Custom replacements configuration file '{}' does not exist. ".format(file_name) +
                    refer_to_faq_msg(69))
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except Exception as exc:
        raise Error("Failed to parse custom replacements configuration file '{}': {}. ".format(file_name, exc) +
                    refer_to_faq_msg(70)) from exc

    result = list()
    validation_errors = list()
    for attrs in data:
        if 'id' not in attrs:
            raise Error('One of the custom replacements in the configuration file "{}" does not contain attribute '
                        '"id". '.format(file_name) +
                        refer_to_faq_msg(71))
        if 'match_kind' not in attrs:
            raise Error('One of the custom replacements in the configuration file "{}" does not contain attribute '
                        '"match_kind". Possible values are "points", "scope" and "general". '.format(file_name) +
                        refer_to_faq_msg(71))
        desc = CustomReplacementDescriptor.create_instance(attrs['match_kind'], attrs['id'], attrs)
        validation_errors.extend(desc.validate_data())
        result.append(desc)
    if len(validation_errors) > 0:
        raise Error("File '{}' validation failed:\n{}. ".format(file_name, "\n".join(validation_errors)) +
                    refer_to_faq_msg(72))
    return result


def generate_pattern_for_node(graph: Graph, sub_graph_pattern: str, node_name: str):
    if sub_graph_pattern == '':
        return node_name
    node_name_components = node_name.split("/")
    cur_name = ''
    matched_index = None  # index of the node name component to start new pattern from
    compiled_pattern = compile(sub_graph_pattern)
    for index in range(0, len(node_name_components)):
        cur_name += node_name_components[index] + "/"
        if match(compiled_pattern, cur_name):
            matched_index = index
            break
    if matched_index is None:
        raise RuntimeError('Node name "{}" does not match pattern "{}"'.format(node_name, sub_graph_pattern))

    if sub_graph_pattern == '' or sub_graph_pattern[-1] != '/':
        sub_graph_pattern += '/'

    sub_graph_nodes = nodes_matching_name_pattern(graph, sub_graph_pattern)
    name_suffix = '/'.join(node_name_components[matched_index + 1:]) + '$'
    if len([node for node in sub_graph_nodes if match(sub_graph_pattern + name_suffix, node)]) == 1:
        return name_suffix

    raise RuntimeError('The pattern that uniquely identifies node "{}" using sub-graph pattern "{}" has not been found'.
                       format(node_name, sub_graph_pattern))
