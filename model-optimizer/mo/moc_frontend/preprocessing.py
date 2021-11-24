# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log

from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg

import numpy as np

from openvino import Function, Layout, PartialShape
from openvino.preprocess import PrePostProcessor


def update_mean_scale_to_dict(input_nodes: list, mean_scale_val, scale):
    """
    Internal function. Updates mean/scale values from array to dictionary
    :param: input_nodes Inputs of model
    :param: mean_scale_val Parsed 'mean_scale_val' object from command line arguments
    :param: scale Global scale factor for all inputs from --scale command line arguments
    """
    if not isinstance(mean_scale_val, dict):
        if len(mean_scale_val) != len(input_nodes):
            raise Error('Numbers of inputs and mean/scale values do not match. ' + refer_to_faq_msg(61))

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


def check_mean_scale_keys(ov_inputs: list, mean_scale_keys: list):
    inputs_used = {}
    for name in mean_scale_keys:
        input_found = False
        for ov_input in ov_inputs:
            if name in ov_input.get_tensor().get_names():
                if ov_input in inputs_used:
                    raise Error('Mean/Scale for {} and {} point to same model input.'
                                .format(name, inputs_used[ov_input]))
                inputs_used[ov_input] = name
                input_found = True
                break
        if not input_found:
            raise Error('Input with name {} wasn\'t found! {}'.format(name, refer_to_faq_msg(83)))


def find_channels_dimension(shape: PartialShape, num_channels: int, name: str, layout_items):
    """
    Internal function. Finds dimension index matching with expected channels number
    Raises error if there is no candidates or number of candidates is > 1
    :param: shape Parameter's partial shape
    :param: num_channels Number of channels to find in shape
    :param: name Parameter's name, used for Error-handling purposes
    :param: layout_items Existing source/target layout items specified by user
    :return: updated layout items with guessed layouts
    """
    if shape.rank.is_dynamic:
        raise Error('Can\'t determine channels dimension for dynamic shape for parameter {}.'
                    .format(name))

    dim_idx_found = -1
    for dim_idx in range(shape.rank.get_length()):
        dim = shape.get_dimension(dim_idx)
        if dim.is_static and dim.get_length() == num_channels:
            if dim_idx_found >= 0:
                raise Error('Can\'t determine channels dimension for {}. '
                            'Input shape is {}, needed channels {}. '
                            'Conflicting dimensions: {} and {}'
                            .format(name, shape, num_channels, dim_idx_found, dim_idx))
            dim_idx_found = dim_idx
    if dim_idx_found < 0:
        raise Error('Can\'t determine channels dimension for {}. '
                    'Input shape is {}, needed channels {}'
                    .format(name, shape, num_channels))
    layout_str = "?" * shape.rank.get_length()
    layout_str = layout_str[:dim_idx_found] + 'C' + layout_str[dim_idx_found+1:]
    layout_items[name] = {
        'source_layout': layout_str,
        'target_layout': None,
        'source_guessed': True
    }
    return layout_items


def guess_source_layouts_by_mean_scale(ov_function: Function, layout_items, mean_scale_values: dict):
    """
    Internal function. Try to guess source layout for input by its shape and/or framework
    :param: ov_function Original model
    :param: layout_items Existing source/target layout items specified by user
    :param: mean_scale_values Dictionary with mean/scale values defined for each argument
    :return: updated layout items with guessed layouts
    """
    # Test code
    # layout_items = {'inputX1': {'source_layout': '?c...', 'target_layout': 'nhwc', 'source_guessed': True},
    #                 'inputX2': {'source_layout': '???c', 'target_layout': None, 'source_guessed': True}
    #                 }
    check_mean_scale_keys(ov_inputs=ov_function.inputs, mean_scale_keys=mean_scale_values.keys())
    for ms_name, mean_scale in mean_scale_values.items():
        num_channels_mean = len(mean_scale['mean']) if mean_scale['mean'] is not None else 0
        num_channels_scale = len(mean_scale['scale']) if hasattr(mean_scale['scale'], '__len__') else 0

        if num_channels_mean > 1 and \
                num_channels_scale > 1 and \
                num_channels_mean is not num_channels_scale:
            raise Error('Mean/Scale values for {} have different sizes: {} {}'
                        .format(ms_name, num_channels_mean, num_channels_scale))

        need_guess_channels = num_channels_mean > 1 or num_channels_scale > 1
        if not need_guess_channels:  # Mean/scale is complex and needs 'channels' specified in layout
            continue

        num_channels = num_channels_mean if num_channels_mean > 1 else num_channels_scale
        for ov_input in ov_function.inputs:
            if ms_name not in ov_input.get_tensor().get_names():
                continue

            layout_exists = False
            layout_item = None
            for name in ov_input.get_tensor().get_names():
                if name in layout_items:
                    layout_item = layout_items[name]
                    if 'source_layout' in layout_item and layout_item['source_layout'] is not None:
                        layout_exists = True
                    break

            if layout_item is not None:
                # If user specified 'target_layout' but didn't specify 'source_layout', skip guessing
                continue

            if not layout_exists:
                layout_items = find_channels_dimension(shape=ov_input.get_partial_shape(),
                                                       num_channels=num_channels,
                                                       name=ms_name,
                                                       layout_items=layout_items)
    log.debug('Layout items after guess: {}'.format(layout_items))
    return layout_items


def guess_source_layouts_for_reverse_channels(ov_function: Function, layout_items):
    """
    Internal function. Try to guess source layout for input by finding dimension with size=3 (RGB/BGR)
    :param: ov_function Original model
    :param: layout_items Existing source/target layout items specified by user
    :return: updated layout items with guessed layouts
    """
    for ov_input in ov_function.inputs:
        layout_exists = False
        layout_item = None
        first_name = list(ov_input.get_tensor().get_names())[0]
        for name in ov_input.get_tensor().get_names():
            if name in layout_items:
                layout_item = layout_items[name]
                if 'source_layout' in layout_item and layout_item['source_layout'] is not None:
                    layout_exists = True
                break

        if layout_item is not None:
            # If user specified 'target_layout' but didn't specify 'source_layout', skip guessing
            continue

        if not layout_exists:
            layout_items = find_channels_dimension(shape=ov_input.get_partial_shape(),
                                                   num_channels=3,
                                                   name=first_name,
                                                   layout_items=layout_items)

    log.error('Layout items after guess: {}'.format(layout_items))
    return layout_items


def apply_preprocessing(ov_function: Function, argv: argparse.Namespace):
    """
    Applies pre-processing of model inputs by adding appropriate operations
    On return, 'ov_function' object will be updated
    :param: ov_function OV function for applying mean/scale pre-processing
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
    layout_items = {}
    # Test code, real values should be taken from argv.layouts
    # layout_items = {'inputX1': {'source_layout': None, 'target_layout': 'nhwc'},
    #                 'inputX2': {'source_layout': None, 'target_layout': None}
    #                 }

    layout_items = guess_source_layouts_by_mean_scale(ov_function, layout_items, mean_scale_values)
    need_reverse = 'reverse_input_channels' in argv and argv.reverse_input_channels
    if need_reverse:
        layout_items = guess_source_layouts_for_reverse_channels(ov_function=ov_function,
                                                                 layout_items=layout_items)

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
        log.debug('Mean/Scale pre-processing applied to {}'.format(node_name))

    # Apply reverse_input_channels
    if need_reverse:
        guess_source_layouts_for_reverse_channels(ov_function=ov_function, layout_items=layout_items)
        for ov_input in ov_function.inputs:
            node_name = list(ov_input.get_tensor().get_names())[0]
            prep.input(node_name).preprocess().reverse_channels()
            log.debug('reverse_input_channels pre-processing applied to {}'.format(node_name))

    # Apply pre-processing builder to a function
    ov_function = prep.build()

    # Remove guessed layout values from ov_function (these values shall not be serialized to IR
    for node_name, layout_values in layout_items.items():
        if 'source_guessed' in layout_values and \
                layout_values['source_guessed'] and \
                layout_values['target_layout'] is None:
            # search for parameter object
            for idx, ov_input in enumerate(ov_function.inputs):
                if node_name in ov_input.get_tensor().get_names():
                    log.debug('Clearing guessed layout {} for {}'
                              .format(layout_values['source_layout'], node_name))
                    ov_function.get_parameters()[idx].layout = Layout()
