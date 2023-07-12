# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
from copy import copy

from openvino.preprocess import PrePostProcessor  # pylint: disable=no-name-in-module,import-error
# pylint: disable=no-name-in-module,import-error
from openvino.runtime import Model, Layout, PartialShape, layout_helpers

from openvino.tools.mo.moc_frontend.layout_utils import update_layout_to_dict
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


def update_mean_scale_to_dict(input_nodes: list, mean_scale_val, scale):
    """
    Internal function. Updates mean/scale values from array to dictionary
    :param: input_nodes Inputs of model
    :param: mean_scale_val Parsed 'mean_scale_val' object from command line arguments
    :param: scale Global scale factor for all inputs from scale command line arguments
    """
    if not isinstance(mean_scale_val, dict):
        if len(mean_scale_val) != len(input_nodes):
            raise Error('Numbers of inputs and mean/scale values do not match. ' + refer_to_faq_msg(61))
        data = copy(mean_scale_val)
        mean_scale_val = {}
        for idx, node in enumerate(input_nodes):
            names_list = list(node.get_tensor().get_names())
            names_list.sort()
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
            names_list.sort()
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


def check_keys_valid(ov_function: Model, dict_to_validate: dict, search_outputs: bool):
    """
    Internal function: checks if keys from cmd line arguments correspond to ov_function's inputs/outputs
    Throws if some key is not found
    Throws if some different keys point to the same actual input/output
    """
    nodes_used = {}
    nodes = ov_function.inputs
    if search_outputs:
        nodes += ov_function.outputs

    # We need to replace all node names from dict to tensor names
    rename_dict = {}
    # Find names for replacing
    for name in dict_to_validate.keys():
        for ov_node in nodes:
            if name in ov_node.get_tensor().get_names():
                break
            elif name == ov_node.get_node().get_friendly_name():
                assert len(ov_node.get_tensor().get_names()) > 0, 'Node must have at least one tensor name'
                new_name = list(ov_node.get_tensor().get_names())[0]
                rename_dict[name] = new_name
                break

    # Replace found node names with tensor names
    for name, new_name in rename_dict.items():
        assert name in dict_to_validate, 'Key {} is not in initial dict'.format(name)
        assert new_name not in dict_to_validate, 'Key {} is already in initial dict'.format(new_name)
        dict_to_validate[new_name] = dict_to_validate[name]
        del dict_to_validate[name]

    # validate the dict
    for name in dict_to_validate.keys():
        node_found = False
        for ov_node in nodes:
            if name in ov_node.get_tensor().get_names():
                if ov_node in nodes_used:
                    raise Error('Key for {} and {} point to same model input/output.'
                                .format(name, nodes_used[ov_node]))
                nodes_used[ov_node] = name
                node_found = True
                break

        if not node_found:
            if not search_outputs:
                raise Error('Input with name {} wasn\'t found! {}'.format(name, refer_to_faq_msg(83)))
            else:
                raise Error('Input/Output with name {} wasn\'t found! {}'.format(name, refer_to_faq_msg(83)))


def update_layout_is_input_flag(ov_function: Model, layout_values: dict):
    """
    Internal function: updates layout_values with flag whether each layout belongs to input or to output
    """
    for name, layout_value in layout_values.items():
        layout_value['is_input'] = False
        for ov_input in ov_function.inputs:
            if name in ov_input.get_tensor().get_names():
                layout_value['is_input'] = True
                break
    return layout_values


def find_channels_dimension(shape: PartialShape, num_channels: int, name: str, layout_values):
    """
    Internal function. Finds dimension index matching with expected channels number
    Raises error if there is no candidates or number of candidates is > 1
    :param: shape Parameter's partial shape
    :param: num_channels Number of channels to find in shape
    :param: name Parameter's name, used for Error-handling purposes
    :param: layout_values Existing source/target layout items specified by user
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
                            'Conflicting dimensions: {} and {}. Please specify layout manually.'
                            .format(name, shape, num_channels, dim_idx_found, dim_idx))
            dim_idx_found = dim_idx
    if dim_idx_found < 0:
        raise Error('Can\'t determine channels dimension for {}. '
                    'Input shape is {}, needed channels {}'
                    .format(name, shape, num_channels))

    # Restrict guessed channels index to particular position depending on tensor shape(3d, 4d, 5d)
    if shape.rank.get_length() == 3:
        # CHW or HWC, possible channels index is 0 or 2
        if dim_idx_found != 0 and dim_idx_found != 2:
            raise Error('Can\'t determine channels dimension for 3D input {} (CHW or HWC) with shape {}. '
                        'Please specify layout containing \'C\' channels manually.'.format(name, shape))
    elif shape.rank.get_length() == 4:
        # NCHW or NHWC, possible channels index is 1 or 3
        if dim_idx_found != 1 and dim_idx_found != 3:
            raise Error('Can\'t determine channels dimension for 4D input {} (NCHW or NHWC) with shape {}. '
                        'Please specify layout containing \'C\' channels manually.'.format(name, shape))
    elif shape.rank.get_length() == 5:
        # NCDHW or NDHWC, possible channels index is 1 or 4
        if dim_idx_found != 1 and dim_idx_found != 4:
            raise Error('Can\'t determine channels dimension for 5D input {} (NCDHW or NDHWC) with shape {}. '
                        'Please specify layout containing \'C\' channels manually.'.format(name, shape))
    else:
        raise Error('Can\'t determine channels dimension for {}D input {} with shape {}.'
                    'Please specify layout containing \'C\' channels manually.'
                    .format(shape.rank.get_length(), name, shape))

    layout_str = "?" * shape.rank.get_length()
    layout_str = layout_str[:dim_idx_found] + 'C' + layout_str[dim_idx_found + 1:]
    layout_values[name] = {
        'source_layout': layout_str,
        'target_layout': None,
        'source_guessed': True,
        'is_input': True
    }
    return layout_values


def guess_source_layouts_by_mean_scale(ov_function: Model, layout_values, mean_scale_values: dict):
    """
    Internal function. Try to guess source layout for input by its shape and/or framework
    :param: ov_function Original model
    :param: layout_values Existing source/target layout items specified by user
    :param: mean_scale_values Dictionary with mean/scale values defined for each argument
    :return: updated layout items with guessed layouts
    """
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

        for i in range(0, len(ov_function.inputs)):
            ov_input = ov_function.input(i)

            if not ov_function.get_parameters()[i].layout.empty:
                continue

            if ms_name not in ov_input.get_tensor().get_names():
                continue

            layout_item = None
            for name in ov_input.get_tensor().get_names():
                if name in layout_values:
                    layout_item = layout_values[name]
                    break

            if layout_item is not None:
                # User specified some layout, skip guessing
                continue

            # Guess layout is applicable only when number of channels is '3'
            if num_channels != 3:
                raise Error('Can\'t determine channels dimension for {}. '
                            'When number of mean/scale values is {} (not 3), '
                            'please specify layout for input manually'.format(ms_name, num_channels))

            layout_values = find_channels_dimension(shape=ov_input.get_partial_shape(),
                                                    num_channels=num_channels,
                                                    name=ms_name,
                                                    layout_values=layout_values)
    return layout_values


def check_suitable_for_reverse(layout: Layout, ov_input):
    """
    Internal function. Checks if input with layout is suitable for reversing channels
    :param: layout Existing source/target layout items specified by user
    :param: ov_input Model's input
    :return: True if reverse channels can be applied to input
    """
    if not layout_helpers.has_channels(layout):
        return False
    if ov_input.get_partial_shape().rank.is_dynamic:
        return False

    c_idx = layout_helpers.channels_idx(layout)
    rank = ov_input.get_partial_shape().rank.get_length()
    if c_idx < 0:
        c_idx += rank
    if c_idx >= rank:
        raise Error('Layout {} for input {} is inconsistent with shape {}'.format(
            layout, ov_input.get_tensor().get_any_name(), ov_input.get_partial_shape()))
    c_num = ov_input.get_partial_shape()[c_idx]
    return c_num.is_dynamic or c_num.get_length() == 3


def guess_source_layouts_for_reverse_channels(ov_function: Model, layout_values):
    """
    Internal function. Try to guess source layout for input by finding dimension with size=3 (RGB/BGR)
    Additionally checks existing layouts and detects suitable inputs for reversing of input channels
    :param: ov_function Original model
    :param: layout_values Existing source/target layout items specified by user
    :return: array with suitable parameters for reversing of input channels
    """
    all_params = []
    suitable_params = []
    for i in range(0, len(ov_function.inputs)):
        ov_input = ov_function.input(i)
        param_info = [ov_input.get_tensor().get_any_name(), ov_input.get_partial_shape()]
        all_params.append(param_info)

        if not ov_function.get_parameters()[i].layout.empty:
            if check_suitable_for_reverse(ov_function.get_parameters()[i].layout, ov_input):
                suitable_params.append(param_info)
            continue

        layout_item = None
        first_name = ov_input.get_tensor().get_any_name()
        for name in ov_input.get_tensor().get_names():
            if name in layout_values:
                layout_item = layout_values[name]
                break

        if layout_item is not None:
            # RIC transformation is applied before changing layout so only source_layout
            # should be checked (even is target_layout is also provided)
            if layout_item.get('source_layout'):
                if check_suitable_for_reverse(Layout(layout_item['source_layout']), ov_input):
                    suitable_params.append(param_info)
            continue

        try:
            layout_values = find_channels_dimension(shape=ov_input.get_partial_shape(),
                                                    num_channels=3,
                                                    name=first_name,
                                                    layout_values=layout_values)
        except Error as e:
            log.debug('Reverse input channels guess did not succeed {}'.format(e))
        else:
            layout = layout_values[first_name].get('source_layout')
            if layout and check_suitable_for_reverse(Layout(layout), ov_input):
                suitable_params.append(param_info)

    if not len(suitable_params):
        raise Error('Network has {} inputs overall, but none of them are suitable for input channels reversing.\n'
                    'Suitable for input channel reversing inputs are 4-dimensional with 3 channels (in case of dynamic '
                    'dimensions C channel must be provided in a layout for this input)\nAll inputs: {}'.format(
            len(all_params), all_params))
    elif len(suitable_params) < len(all_params):
        log.error('Network has {} inputs overall, but only {} of them are suitable for input channels reversing.\n'
                  'Suitable for input channel reversing inputs are 4-dimensional with 3 channels (in case of dynamic '
                  'dimensions C channel must be provided in a layout for this input)\nAll inputs: {}\n'
                  'Suitable inputs {}'.format(len(all_params), len(suitable_params), all_params, suitable_params),
                  extra={'is_warning': True})
    return suitable_params


def update_tensor_names_to_first_in_sorted_list(values_dict: dict, ov_function: Model):
    if not isinstance(values_dict, dict):
        return values_dict
    updated_dict = {}
    used_nodes = {}
    for name, value in values_dict.items():
        input_found = False
        for input in ov_function.inputs:
            tensor_names = list(input.names)
            tensor_names.sort()
            if not (name in tensor_names or name == input.node.get_friendly_name()):
                continue
            if input in used_nodes:
                raise Error("Tensor names {} and {} refer to the same node.".format(name, used_nodes[input]))
            used_nodes.update({input: name})
            updated_dict[tensor_names[0]] = value
            input_found = True
            break
        if not input_found:
            raise Error('Input with name {} wasn\'t found! {}'.format(name, refer_to_faq_msg(83)))

    return updated_dict


def apply_preprocessing(ov_function: Model, argv: argparse.Namespace):
    """
    Applies pre-processing of model inputs by adding appropriate operations
    On return, 'ov_function' object will be updated
    Expected 'argv.mean_scale_values' formats examples:
        a) Dict: {'inputName': {'mean': [1., 2., 3.], 'scale': [2., 4., 8.]}}
        b) List: list(np.array([(np.array([1., 2., 3.]), np.array([2., 4., 6.])),
                     (np.array([7., 8., 9.]), np.array([5., 6., 7.])))
    Expected 'argv.layout_values' format examples:
        a) Specific layouts for inputs and outputs
        { 'input1': {
                 'source_layout': 'nchw',
                 'target_layout': 'nhwc'
             },
             'output2': {
                 'source_layout': 'nhwc'
             }
        }
        b) Layout for single input: {'': {'source_layout': 'nchw'}}
    :param: ov_function OV function for applying mean/scale pre-processing
    :param: argv Parsed command line arguments
    """
    prep = PrePostProcessor(ov_function)

    if 'mean_scale_values' in argv and argv.mean_scale_values:
        mean_scale_values = argv.mean_scale_values
    else:
        mean_scale_values = {}

    # mean_scale_values stores mean/scale values from command line with names which were set by user.
    # For models with single input scale or mean may be unnamed, so name is set by first tensor name from
    # names list. This may lead to different naming of preprocessing params for a single node and lead to error.
    # To make naming for mean/scale values unified, names provided by user are renamed here
    # by the first tensor name from sorted names list.
    mean_scale_values = update_tensor_names_to_first_in_sorted_list(mean_scale_values, ov_function)
    mean_scale_values = update_mean_scale_to_dict(input_nodes=ov_function.inputs,
                                                  mean_scale_val=mean_scale_values,
                                                  scale=argv.scale)
    # On return, mean_scale_values is a dictionary with input names as key and mean/scale pair as value
    # {'inputName': {'mean': [1., 2., 3.], 'scale': [2.]}}

    layout_values = {}
    if 'layout_values' in argv and argv.layout_values:
        layout_values = update_layout_to_dict(ov_function.inputs, argv.layout_values,
                                              lambda ov_input: ov_input.get_tensor().get_names())

    check_keys_valid(ov_function=ov_function, dict_to_validate=mean_scale_values, search_outputs=False)
    check_keys_valid(ov_function=ov_function, dict_to_validate=layout_values, search_outputs=True)

    layout_values = update_layout_is_input_flag(ov_function, layout_values)
    layout_values = guess_source_layouts_by_mean_scale(ov_function, layout_values, mean_scale_values)
    need_reverse = 'reverse_input_channels' in argv and argv.reverse_input_channels
    suitable_params_ric = []
    if need_reverse:
        suitable_params_ric = guess_source_layouts_for_reverse_channels(ov_function=ov_function,
                                                                        layout_values=layout_values)

    for node_name, layout_value in layout_values.items():
        if layout_value.get('source_layout'):
            if layout_value.get('is_input'):
                prep.input(node_name).model().set_layout(Layout(layout_value['source_layout']))
            else:
                prep.output(node_name).model().set_layout(Layout(layout_value['source_layout']))
        if layout_value.get('target_layout'):
            if layout_value.get('is_input'):
                prep.input(node_name).tensor().set_layout(Layout(layout_value['target_layout']))
            else:
                prep.output(node_name).tensor().set_layout(Layout(layout_value['target_layout']))

    # Apply reverse_input_channels
    if need_reverse:
        for name, _ in suitable_params_ric:
            prep.input(name).preprocess().reverse_channels()
            log.debug('reverse_input_channels pre-processing applied to {}'.format(name))

    for node_name, node_mean_scale_values in mean_scale_values.items():
        # Apply mean first, then scale
        if node_mean_scale_values['mean'] is not None:
            prep.input(node_name).preprocess().mean(node_mean_scale_values['mean'])
        if node_mean_scale_values['scale'] is not None:
            prep.input(node_name).preprocess().scale(node_mean_scale_values['scale'])
        log.debug('Mean/Scale pre-processing applied to {}'.format(node_name))

    # Apply pre-processing builder to a function
    ov_function = prep.build()

    # Remove guessed layout values from ov_function (these values shall not be serialized to IR
    for node_name, layout_value in layout_values.items():
        if layout_value.get('source_guessed') and \
                not layout_value.get('target_layout'):
            # search for parameter object
            for idx, ov_input in enumerate(ov_function.inputs):
                if node_name in ov_input.get_tensor().get_names():
                    log.debug('Clearing guessed layout {} for {}'
                              .format(layout_value['source_layout'], node_name))
                ov_function.get_parameters()[idx].layout = Layout()
