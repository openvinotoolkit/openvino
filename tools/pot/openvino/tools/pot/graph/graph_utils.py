# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from copy import deepcopy
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph
from openvino.tools.mo.utils.logger import init_logger
from openvino.runtime import Core, serialize  # pylint: disable=E0401,E0611
from openvino.runtime.passes import Manager # pylint: disable=E0401,E0611
from openvino._offline_transformations import apply_pot_transformations # pylint: disable=import-error,no-name-in-module

from ..graph.passes import ModelPreprocessor, remove_converts, add_removed_converts
from ..utils.logger import stdout_redirect
from ..configs.config import GNA_DEVICES

init_logger('ERROR', False)
core = Core()
core.set_property({"ENABLE_MMAP": "NO"})
pass_manager = Manager()


def load_graph(model_config, target_device='ANY'):
    """ Loads model from specified path
    :return NetworkX model
     """
    serialized_bin_path = os.path.join(tempfile.gettempdir(), 'serialized_ir.bin')
    serialized_xml_path = os.path.join(tempfile.gettempdir(), 'serialized_ir.xml')
    bin_path = model_config.weights
    xml_path = model_config.model

    if target_device in GNA_DEVICES:
        model = core.read_model(model=xml_path, weights=bin_path)
        apply_pot_transformations(model, target_device.encode('utf-8'))
        bin_path = serialized_bin_path
        xml_path = serialized_xml_path
        serialize(model, xml_path=xml_path, bin_path=bin_path)

    if not os.path.exists(xml_path):
        raise RuntimeError('Input model xml should link to an existing file. Please, provide a correct path.')

    if not os.path.exists(bin_path):
        raise RuntimeError('Input model bin should link to an existing file. Please, provide a correct path.')

    graph_from_ir, meta_data = stdout_redirect(restore_graph_from_ir, xml_path, bin_path)

    if graph_from_ir.graph['ir_version'] == 10:
        raise AssertionError(
            'POT does not support version 10 of IR.'
            'Please convert the model with the newer version of OpenVINO '
            'or use the POT from OpenVINO 2021.4.2 to work with version 10 of IR.')

    orig_graph_from_ir, meta_data = stdout_redirect(restore_graph_from_ir, model_config.model, model_config.weights)

    meta_data['quantization_parameters'] = model_config.quantization_info
    graph_from_ir.meta_data = meta_data
    graph_from_ir.graph['cmd_params'] = orig_graph_from_ir.graph['cmd_params']
    for_graph_and_each_sub_graph_recursively(graph_from_ir, remove_converts)
    model_preprocessing(graph_from_ir)
    if os.path.exists(serialized_xml_path):
        os.remove(serialized_xml_path)
    if os.path.exists(serialized_bin_path):
        os.remove(serialized_bin_path)
    return graph_from_ir


def save_graph(graph: Graph, save_path, model_name=None):
    """ Save model as IR in specified path
    :param graph: NetworkX model to save
    :param save_path: path to save the model
    :param model_name: name under which the model will be saved
     """
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except PermissionError as e:
            raise type(e)(
                'Failed to create a directory {}. Permission denied. '.format(save_path))
    else:
        if not os.access(save_path, os.W_OK):
            raise PermissionError(
                'Output directory {} is not writable for the current user. '.format(save_path))
    graph_copy = deepcopy(graph)
    add_removed_converts(graph_copy)
    save_restored_graph(graph=graph_copy, path=save_path, meta_data=graph.meta_data,
                        name=model_name)


def model_preprocessing(model):
    ModelPreprocessor().find_and_replace_pattern(model)
    model.clean_up()
