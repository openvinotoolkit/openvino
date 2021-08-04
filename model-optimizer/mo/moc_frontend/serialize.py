# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import argparse
import os
from mo.front.common.layout import get_features_dim
from mo.pipeline.common import get_ir_version
from mo.back.ie_ir_ver_2.emitter import append_ir_info
from mo.utils.cli_parser import get_meta_info
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg

# pylint: disable=no-name-in-module,import-error
from ngraph import Function
from ngraph import function_to_cnn
from openvino.inference_engine import IENetwork
from openvino.offline_transformations import ApplyScaleInputs,\
    ApplySubtractMeanInputs, ConstantInfo


def apply_mean_scale(network: IENetwork, input_nodes, preprocessing_name: str, mean_scale_val):
    is_mean = preprocessing_name == 'mean'
    print("Inputs={}".format(input_nodes))
    if not isinstance(mean_scale_val, dict):
        # TODO: The case when input names to apply mean/scales weren't specified
        if len(mean_scale_val) != len(input_nodes):
            print("Values [{}] {}, nodes={} {}".format(
                mean_scale_val, len(mean_scale_val), input_nodes, len(input_nodes)))
            raise Error('Numbers of inputs and mean/scale values do not match. ' + refer_to_faq_msg(61))

        data = np.copy(mean_scale_val)
        mean_scale_val = {}
        for idx, node in enumerate(input_nodes):
            mean_scale_val.update(
                {
                    node.soft_get('name', node.id): {
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


def moc_emit_ir(ngraph_function: Function, argv: argparse.Namespace):
    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()

    network = function_to_cnn(ngraph_function)

    # Add mean/scale
    if argv.mean_scale_values:
        params = ngraph_function.get_parameters()
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

    print("Done")





    orig_model_name = os.path.normpath(os.path.join(output_dir, argv.model_name))
    network.serialize(orig_model_name + ".xml", orig_model_name + ".bin")

    del argv.feManager

    # add meta information to IR
    append_ir_info(file=orig_model_name,
                   meta_info=get_meta_info(argv),
                   mean_data=None,
                   input_names=None)

    print('[ SUCCESS ] Generated IR version {} model.'.format(get_ir_version(argv)))
    print('[ SUCCESS ] XML file: {}.xml'.format(orig_model_name))
    print('[ SUCCESS ] BIN file: {}.bin'.format(orig_model_name))
    return 0
