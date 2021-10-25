# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
from mo.graph.graph import Node

from . import editor as ge, builder as gb
from .nx_model import NXModel
from .passes import compress_weights
from ..utils.utils import convert_output_key


def load_model(model_config, target_device='ANY'):
    """ Loads model from specified path
    :return NXModel instance
    """
    return NXModel(config=model_config, target_device=target_device)


def save_model(model: NXModel, save_path, model_name=None, for_stat_collection=False):
    """ Save model as IR in specified path
    :param model: NXModel instance to save
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
    for model_dict in models:
        node_names_ = node_names if len(models) == 1 \
            else [node_name for node_name in node_names
                  if convert_output_key(node_name).startswith(model_dict['name'])]
        model_dict['model'].add_outputs(node_names_)


def compress_model_weights(model: NXModel):
    """Apply transformations to save model weights to INT8."""
    for model_dict in model.models:
        compress_weights(model_dict['model'])


def get_nodes_by_type(model: NXModel, types: list):
    """ Returns all nodes with type from types collection
    :param model: NXModel model
    :param types: list of required types
    :return list of nodes filtered by 'types' collection
    """
    return [node for model_dict in model.models
            for node in ge.get_nodes_by_type(model_dict['model'], types)]


def get_node_by_name(model: NXModel, name: str) -> Node:
    """ Returns node by name
    :param model: NXModel model
    :param name: name of the node
    :return node from model (of type Node or None if there's no such node)
    """
    names = [ge.get_node_by_name(model_dict['model'], name)
             for model_dict in model.models]
    names = [name for name in names if name is not None]
    if len(names) > 1:
        raise RuntimeError('The model contains more than one node with the name {}'.format(name))

    return names[0] if names else None


def get_all_operation_nodes(model: NXModel):
    """ Returns sequence of all nodes in all graphs
    :param model: NXModel model
    :return list of all nodes
    """
    return [node for model_dict in model.models
            for node in ge.get_all_operation_nodes(model_dict['model'])]


def build_model_for_node(nx_model, input_name, input_shape, node, remove_bias=False,
                         remove_fake_quantize=False, target_device='ANY'):
    """ Build Model containing Subgraph of NXModel (input - node - output).
    The Convolution, FullyConnected node types are supported.
    :param nx_model: NXModel model
    :param input_name: name of the input node in the generated graph
    :param input_shape: shape of the input node in the generated graph
    :param node: node for which graph (input - node - output) will be generated
    :param remove_bias: remove bias in the generated graph
    :param remove_fake_quantize: remove fake quantize nodes in the generated graph
    :param target_device: device for processing
    :return: generated NXModel instance.
    """
    candidates = [model_dict['model'] for model_dict in nx_model.models
                  if ge.get_node_by_name(model_dict['model'], input_name)]
    if len(candidates) > 1:
        raise RuntimeError('Name collision: {}'.format(input_name))
    model = candidates[0]
    op_graph = gb.build_graph_for_node(model, input_name, input_shape, node, remove_bias, remove_fake_quantize)
    return NXModel(graph=op_graph, target_device=target_device)


def models_union(first_model, second_model):
    """ Return the union of NXModel models
    :return NXModel instance - union of first_model and second_model
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
