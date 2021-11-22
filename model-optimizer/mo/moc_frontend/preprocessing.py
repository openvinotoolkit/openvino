# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from mo.front.common.layout import get_features_dim
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg

# pylint: disable=no-name-in-module,import-error
#from ngraph import Function

import numpy as np

from openvino.pyopenvino.preprocess import PrePostProcessor,\
    InputInfo, InputTensorInfo, PreProcessSteps
from openvino import Function


def update_mean_scale_to_dict(input_nodes: list, mean_scale_val, scale):
    """
    Internal function. Updates mean/scale values from array to dictionary
    :param: input_nodes Parameters of input model
    :param: mean_scale_val Parsed 'mean_scale_val' object from command line arguments
    """
    if not isinstance(mean_scale_val, dict):
        if len(mean_scale_val) != len(input_nodes):
            raise Error('Numbers of inputs and mean/scale values do not match. ')

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

    if scale:
        for idx, node in enumerate(input_nodes):
            old_val = mean_scale_val[node.get_friendly_name()] if node.get_friendly_name() in mean_scale_val else None
            mean_scale_val.update(
                {
                    node.get_friendly_name(): {
                        'mean': old_val['mean'] if old_val and 'mean' in old_val else None,
                        'scale': scale
                    }
                }
            )
    return mean_scale_val


def add_scale_to_dict(input_nodes: list, mean_scale_val: dict, scale):
    """
    Internal function. Adds global 'scale' to each input node
    :param: input_nodes Parameters of input model
    :param: mean_scale_val Parsed 'mean_scale_val' object from command line arguments
    """
    if not isinstance(mean_scale_val, dict):
        if len(mean_scale_val) != len(input_nodes):
            raise Error('Numbers of inputs and mean/scale values do not match. ')

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
    if scale:
        for idx, node in enumerate(input_nodes):
            mean_scale_val.update(
                {
                    node.get_friendly_name(): {
                        'mean': mean_scale_val[node.get_friendly_name()]['mean'],
                        'scale': scale
                    }
                }
            )

    return mean_scale_val

#
#     converted_map = {}
#     for node_name, node_mean_scale_values in mean_scale_val.items():
#         found_node = None
#         value = node_mean_scale_values['mean'] if is_mean else node_mean_scale_values['scale']
#         if value is None:
#             continue
#
#         for node in input_nodes:
#             if node.get_friendly_name() == node_name:
#                 found_node = node
#                 break
#
#         if found_node is None:
#             raise Error('Input with name {} wasn\'t found!'.format(node_name) +
#                         refer_to_faq_msg(83))
#
#         node_rank = found_node.get_partial_shape().rank.get_length()
#         features_dim_idx = 0
#         if value.size != 1:
#             # TODO: layout shall be specified in future via command line arguments
#             features_dim_idx = get_features_dim('NCHW', node_rank)
#
#         features_dim = found_node.get_partial_shape().get_dimension(features_dim_idx)
#         assert value.size == features_dim.get_length() or value.size == 1
#         converted_map[node_name] = ConstantInfo(data=value, axis=features_dim_idx, shape_size=node_rank)
#     if is_mean:
#         ApplySubtractMeanInputs(network=network, values=converted_map)
#     else:
#         ApplyScaleInputs(network=network, values=converted_map)


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


def apply_preprocessing(ov_function: Function, argv: argparse.Namespace):
    """
    Applies preprocessing of network inputs by adding appropriate operations
    On return, 'ov_function' object will be updated
    :param: ov_function OV function for applying mean/scale preprocessing
    :param: argv Parsed command line arguments
    """
    prep = PrePostProcessor(ov_function)

    params = ov_function.get_parameters()

    # TODO: set source/target layout from argv options (Issue 55816)

    # Apply mean/scale
    if argv.mean_scale_values:
        mean_scale_values = argv.mean_scale_values
    else:
        mean_scale_values = {}

    mean_scale_values = update_mean_scale_to_dict(input_nodes=params,
                                                  mean_scale_val=mean_scale_values,
                                                  scale=argv.scale)

    for node_name, node_mean_scale_values in mean_scale_values.items():
        steps = PreProcessSteps()
        if node_mean_scale_values['mean'] is not None:
            steps.mean(node_mean_scale_values['mean'])
        if node_mean_scale_values['scale'] is not None:
            steps.scale(node_mean_scale_values['scale'])
        prep.input(InputInfo(node_name).preprocess(steps)) # TODO: use getters API

    prep.build()
