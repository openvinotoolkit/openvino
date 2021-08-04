# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import argparse
from mo.front.common.layout import get_features_dim
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg

# pylint: disable=no-name-in-module,import-error
from ngraph import Function
from openvino.inference_engine import IENetwork
from openvino.offline_transformations import ApplyScaleInputs,\
    ApplySubtractMeanInputs, ConstantInfo


def apply_mean_scale(network: IENetwork, input_nodes: list, preprocessing_name: str, mean_scale_val):
    is_mean = preprocessing_name == 'mean'
    print("Inputs={}, type={}".format(input_nodes, type(mean_scale_val)))
    if not isinstance(mean_scale_val, dict):
        # TODO: The case when input names to apply mean/scales weren't specified
        if len(mean_scale_val) != len(input_nodes):
            print("Values [{}] {}, nodes={} {}".format(
                mean_scale_val, len(mean_scale_val), input_nodes, len(input_nodes)))
            raise Error('Numbers of inputs and mean/scale values do not match. ' + refer_to_faq_msg(61))

        print("Inside if")
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
        print("node_name={}, vals={}".format(node_name, node_mean_scale_values))
        found_node = None
        value = node_mean_scale_values['mean'] if is_mean else node_mean_scale_values['scale']
        if value is None:
            continue

        for node in input_nodes:
            print("Checking: {}".format(node.get_friendly_name()))
            if node.get_friendly_name() == node_name:
                print("Found {}".format(node_name))
                found_node = node
                break

        if found_node is None:
            raise Error('Input with name {} wasn\'t found!'.format(node_name) +
                        refer_to_faq_msg(83))

        node_rank = found_node.get_partial_shape().rank.get_length()
        features_dim_idx = get_features_dim('NCHW', node_rank)
        print("dim_idx={}, shape_len={}".format(features_dim_idx, node_rank))
        features_dim = found_node.get_partial_shape().get_dimension(features_dim_idx)
        assert value.size == features_dim.get_length() or value.size == 1
        converted_map[node_name] = ConstantInfo(data=value, axis=features_dim_idx, shape_size=node_rank)
    print("conv map = {}".format(converted_map))
    if is_mean:
        ApplySubtractMeanInputs(network=network, values=converted_map)
    else:
        ApplyScaleInputs(network=network, values=converted_map)


def construct_mean_scale_val(input_nodes: list, preprocessing_name: str, value: float):
    result = {}
    for input_node in input_nodes:
        result[input_node.get_friendly_name()] = {preprocessing_name: np.array([value])}

    print("Constructed val: {}".format(result))
    return result


def process_mean_scale(network: IENetwork, ngraph_function: Function, argv: argparse.Namespace):

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
        print("Provided mean/scale val: {}".format(values))
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
