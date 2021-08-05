# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from mo.front.common.layout import get_features_dim
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg

# pylint: disable=no-name-in-module,import-error
from ngraph import Function

import numpy as np

from openvino.inference_engine import IENetwork
from openvino.offline_transformations import ApplyScaleInputs,\
    ApplySubtractMeanInputs, ConstantInfo


def apply_mean_scale(network: IENetwork, input_nodes: list, preprocessing_name: str, mean_scale_val):
    """
    Internal function. Applies mean or scale transformation to 'IE network'. On return 'network' object will be updated
    Logic is similar to AddMeanScaleValues.find_and_replace_pattern adopted for MOC objects
    :param: input_nodes Parameters of input model
    :param: network Inference Engine Network object
    :param: preprocessing_name Name of pre-processing. Can be either 'mean' or 'scale'
    :param: mean_scale_val Parsed 'mean_scale_val' object from command line arguments
    """
    is_mean = preprocessing_name == 'mean'
    if not isinstance(mean_scale_val, dict):
        if len(mean_scale_val) != len(input_nodes):
            raise Error('Numbers of inputs and mean/scale values do not match. ' + refer_to_faq_msg(61))

        data = np.copy(mean_scale_val)
        mean_scale_val = {}
        for idx, node in enumerate(input_nodes):
            mean_scale_val.update(
                {
                    node.get_friendly_name(): {
                        'mean': data[idx][0],
                        'scale': data[idx][1]
                    }
                }
            )

    converted_map = {}
    for node_name, node_mean_scale_values in mean_scale_val.items():
        found_node = None
        value = node_mean_scale_values['mean'] if is_mean else node_mean_scale_values['scale']
        if value is None:
            continue

        for node in input_nodes:
            if node.get_friendly_name() == node_name:
                found_node = node
                break

        if found_node is None:
            raise Error('Input with name {} wasn\'t found!'.format(node_name) +
                        refer_to_faq_msg(83))

        node_rank = found_node.get_partial_shape().rank.get_length()
        features_dim_idx = 0
        if value.size != 1:
            # TODO: layout shall be specified in future via command line arguments
            features_dim_idx = get_features_dim('NCHW', node_rank)

        features_dim = found_node.get_partial_shape().get_dimension(features_dim_idx)
        assert value.size == features_dim.get_length() or value.size == 1
        converted_map[node_name] = ConstantInfo(data=value, axis=features_dim_idx, shape_size=node_rank)
    if is_mean:
        ApplySubtractMeanInputs(network=network, values=converted_map)
    else:
        ApplyScaleInputs(network=network, values=converted_map)


def construct_mean_scale_val(input_nodes: list, preprocessing_name: str, value: float):
    """
    Internal function. Constructs 'mean_scale_val' object based on single value provided
    :param: input_nodes Parameters/Inputs of input model
    :param: preprocessing_name Name of pre-processing. Can be either 'mean' or 'scale'
    :param: value Single value representing scale factor or mean constant
    :return: Constructed 'mean_scale_val' object suitable for apply_mean_scale
    """
    result = {}
    for input_node in input_nodes:
        result[input_node.get_friendly_name()] = {preprocessing_name: np.array([value])}
    return result


def process_mean_scale(network: IENetwork, ngraph_function: Function, argv: argparse.Namespace):
    """
    Performs mean/scale preprocessing transformation of network inputs by adding appropriate operations
    On return, 'network' and 'ngraph_function' objects will be updated
    Both IENetwork and Function are provided to reduce conversion overhead
    :param: network IENetwork object converted from 'ngraph_function'
    :param: ngraph_function nGraph function related to 'network'
    :param: argv Parsed command line arguments
    """
    params = ngraph_function.get_parameters()
    if argv.scale:
        scale_val = construct_mean_scale_val(input_nodes=params,
                                             preprocessing_name='scale',
                                             value=argv.scale)
        apply_mean_scale(network=network,
                         input_nodes=params,
                         preprocessing_name='scale',
                         mean_scale_val=scale_val)

    # Add mean/scale
    if argv.mean_scale_values:
        values = argv.mean_scale_values
        # Add 'scale' first right after inputs
        apply_mean_scale(network=network,
                         input_nodes=params,
                         preprocessing_name='scale',
                         mean_scale_val=values)
        # Add 'mean' right after inputs
        # Total graph will be input->mean->scale
        apply_mean_scale(network=network,
                         input_nodes=params,
                         preprocessing_name='mean',
                         mean_scale_val=values)
