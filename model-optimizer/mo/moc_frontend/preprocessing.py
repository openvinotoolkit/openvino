# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log

from mo.front.common.layout import get_features_dim
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg

import numpy as np

from openvino.preprocess import PrePostProcessor, InputInfo, InputTensorInfo, PreProcessSteps
from openvino import Function, Layout


def update_mean_scale_to_dict(input_nodes: list, mean_scale_val, scale):
    """
    Internal function. Updates mean/scale values from array to dictionary
    :param: input_nodes Inputs of model
    :param: mean_scale_val Parsed 'mean_scale_val' object from command line arguments
    :param: scale Global scale factor for all inputs from --scale command line arguments
    """
    if not isinstance(mean_scale_val, dict):
        if len(mean_scale_val) != len(input_nodes):
            raise Error('Numbers of inputs and mean/scale values do not match. ')

        data = np.copy(mean_scale_val)
        mean_scale_val = {}
        for idx, node in enumerate(input_nodes):
            names_list = list(node.get_tensor().get_names())
            if not names_list:
                continue
            node_name = names_list[0]
            mean_scale_val.update(
                {
                    node_name: {
                        'mean': data[idx][0],
                        'scale': data[idx][1]
                    }
                }
            )

    if scale:
        for node in input_nodes:
            names_list = list(node.get_tensor().get_names())
            if not names_list:
                continue
            node_name = names_list[0]
            old_val = mean_scale_val[node_name] if node_name in mean_scale_val else None
            mean_scale_val.update(
                {
                    node_name: {
                        'mean': old_val['mean'] if old_val and 'mean' in old_val else None,
                        'scale': scale
                    }
                }
            )
    return mean_scale_val


def guess_source_layouts(ov_function: Function, layout_items, argv: argparse.Namespace):
    """
    Internal function. Try to guess source layout for input by its shape and/or framework
    :param: ov_function Original model
    :param: layout_items Existing source/target layout items specified by user
    :param: argv Parsed command line arguments
    :return: updated layout items with guessed layouts
    """
    # Test code
    layout_items = {'inputX1': {'source_layout': 'nchw', 'target_layout': 'nhwc', 'source_guessed': True},
                    'inputX2': {'source_layout': 'nhwc', 'target_layout': None, 'source_guessed': True}
                    }
    return layout_items

def apply_preprocessing(ov_function: Function, argv: argparse.Namespace):
    """
    Applies preprocessing of network inputs by adding appropriate operations
    On return, 'ov_function' object will be updated
    :param: ov_function OV function for applying mean/scale preprocessing
    :param: argv Parsed command line arguments
    """
    prep = PrePostProcessor(ov_function)

    if argv.mean_scale_values:
        mean_scale_values = argv.mean_scale_values
    else:
        mean_scale_values = {}

    mean_scale_values = update_mean_scale_to_dict(input_nodes=ov_function.inputs,
                                                  mean_scale_val=mean_scale_values,
                                                  scale=argv.scale)

    # TODO: construct layout_items from argv options (Issue 55816)
    layout_items = {}  # Should be argv.layouts
    # Test code, real values should be taken from argv.layouts
    layout_items = {'inputX1': {'source_layout': None, 'target_layout': 'nhwc'},
                    'inputX2': {'source_layout': None, 'target_layout': None}
                    }

    layout_items = guess_source_layouts(ov_function, layout_items, argv)
    for node_name, layout_values in layout_items.items():
        if layout_values['source_layout'] is not None:
            prep.input(node_name).network().set_layout(Layout(layout_values['source_layout']))
        if layout_values['target_layout'] is not None:
            prep.input(node_name).tensor().set_layout(Layout(layout_values['target_layout']))

    for node_name, node_mean_scale_values in mean_scale_values.items():
        # Apply mean first, then scale
        if node_mean_scale_values['mean'] is not None:
            prep.input(node_name).preprocess().mean(node_mean_scale_values['mean'])
        if node_mean_scale_values['scale'] is not None:
            prep.input(node_name).preprocess().scale(node_mean_scale_values['scale'])
        log.debug('Mean/Scale preprocessing applied to {}'.format(node_name))

    # Apply reverse-input-channels
    if argv.reverse_input_channels:
        for ov_input in ov_function.inputs:
            node_name = list(ov_input.get_tensor().get_names())[0]
            prep.input(node_name).preprocess().reverse_channels()
            log.debug('reverse_input_channels preprocessing applied to {}'.format(node_name))

    # Apply preprocessing builder to a function
    ov_function = prep.build()

    # Remove guessed layout values from ov_function (these values shall not be serialized to IR
    for node_name, layout_values in layout_items.items():
        if 'source_guessed' in layout_values and layout_values['source_guessed'] and layout_values['target_layout'] is None:
            # search for parameter object
            for idx, input in enumerate(ov_function.inputs):
                if node_name in input.get_tensor().get_names():
                    log.debug('Clearing guessed layout {} for {}'.format(layout_values['source_layout'], node_name))
                    ov_function.get_parameters()[idx].set_layout(Layout())
