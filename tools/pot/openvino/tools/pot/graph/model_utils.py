# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.infer import type_infer
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively

from . import editor as ge, builder as gb
from .nx_model import CompressedModel
from .passes import compress_weights
from ..utils.utils import convert_output_key


def load_model(model_config, target_device='ANY'):
    """ Loads model from specified path
    :return CompressedModel instance
    """
    return CompressedModel(config=model_config, target_device=target_device)


def save_model(model: CompressedModel, save_path, model_name=None, for_stat_collection=False):
    """ Save model as IR in specified path
    :param model: CompressedModel instance to save
    :param save_path: path to save the model
    :param model_name: name under which the model will be saved
    :param for_stat_collection: whether model is saved to be used
    for statistic collection or for normal inference (affects only cascaded models)
    :return model_paths: list of dictionaries:
    'name': model name (for cascade models only)
    'model': path to xml
    'weights': path to bin
    """
    model_paths = model.save(save_path, model_name=model_name, for_stat_collection=for_stat_collection)
    return model_paths


def add_outputs(models, node_names):
    """ Applies add_outputs to each model in models
    param models: list of dictionaries
    'name': model name (for cascaded models only)
    'model': IE model instance
    """
    outputs_per_model = {}
    for model_dict in models:
        model_name = model_dict['model'].friendly_name
        model_node_names = list(node_names[model_name].values())
        node_names_ = model_node_names if len(models) == 1 \
            else [node_name for node_name in model_node_names
                  if convert_output_key(node_name).startswith(model_dict['name'])]
        outputs = model_dict['model'].add_outputs(node_names_)
        outputs_per_model[model_name] = (outputs if outputs else [])
    return outputs_per_model


def compress_model_weights(model: CompressedModel):
    """Apply transformations to save model weights to INT8."""
    for model_dict in model.models:
        for_graph_and_each_sub_graph_recursively(model_dict['model'], compress_weights)


# TODO: set recursively = True to enable subgraphs quantization
def get_nodes_by_type(model: CompressedModel, types: list, recursively: bool = True):
    """ Returns all nodes with type from types collection
    :param model: CompressedModel model
    :param types: list of required types
    :param recursively: whether return all nodes from the model
    and each subgraph or only from the external model
    :return list of nodes filtered by 'types' collection
    """
    return [node for model_dict in model.models
            for node in ge.get_nodes_by_type(model_dict['model'], types, recursively)]


def get_node_by_name(model: CompressedModel, name: str) -> Node:
    """ Returns node by name found in the graph and each subgraph
    :param model: CompressedModel model
    :param name: name of the node
    :return node from model (of type Node or None if there's no such node)
    """
    names = [ge.get_node_by_name(model_dict['model'], name)
             for model_dict in model.models]
    names = [name for name in names if name is not None]
    if len(names) > 1:
        raise RuntimeError('The model contains more than one node with the name {}'.format(name))

    return names[0] if names else None


# TODO: set recursively = True to enable subgraphs quantization
def get_all_operation_nodes(model: CompressedModel, recursively: bool = True):
    """ Returns sequence of all nodes in all graphs
    :param model: CompressedModel model
    :param recursively: whether return all nodes from the model
    and each subgraph or only from the external model
    :return list of all nodes
    """
    return [node for model_dict in model.models
            for node in ge.get_all_operation_nodes(model_dict['model'], recursively)]


def build_model_for_node(nx_model, input_name, input_shape, node, remove_bias=False,
                         remove_fake_quantize=False, target_device='ANY'):
    """ Build Model containing Subgraph of CompressedModel (input - node - output).
    The Convolution, MatMul node types are supported.
    :param nx_model: CompressedModel model
    :param input_name: name of the input node in the generated graph
    :param input_shape: shape of the input node in the generated graph
    :param node: node for which graph (input - node - output) will be generated
    :param remove_bias: remove bias in the generated graph
    :param remove_fake_quantize: remove fake quantize nodes in the generated graph
    :param target_device: device for processing
    :return: generated CompressedModel instance.
    """
    candidates = [model_dict['model'] for model_dict in nx_model.models
                  if ge.get_node_by_name(model_dict['model'], input_name)]
    if len(candidates) > 1:
        raise RuntimeError('Name collision: {}'.format(input_name))
    model = candidates[0]
    op_graph = gb.build_graph_for_node(model, input_name, input_shape, node, remove_bias, remove_fake_quantize)
    return CompressedModel(graph=op_graph, target_device=target_device)


def models_union(first_model, second_model):
    """ Return the union of CompressedModel models
    :return CompressedModel instance - union of first_model and second_model
    """
    union = first_model
    union_models = union.models
    for model_dict, model_dict_2, union_dict in zip(first_model.models, second_model.models, union_models):
        model_1 = model_dict['model']
        model_2 = model_dict_2['model']
        union_dict['model'] = nx.union(model_1, model_2)
        union_dict['model'].graph.update(model_1.graph)
        union_dict['model'].graph.update(model_2.graph)
    return union

def nx_type_infer(model):
    """ Apply type_infer for each model in CompressedModel wrapper
    """
    for model_dict in model.models:
        type_infer(model_dict['model'])
