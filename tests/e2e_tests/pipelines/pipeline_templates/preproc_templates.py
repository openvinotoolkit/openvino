# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessing templates processed by testing framework.
"""

from collections import OrderedDict


def assemble_preproc(batch=None,
                     h=None,
                     w=None,
                     resize_mode="nearest",
                     mean=None,
                     mean_file=None,
                     scale_factor=None,
                     normalization_factor=None,
                     reverse_channels=None,
                     permute_order=None,
                     expand_dims=True,
                     target_layers=None,
                     layers_to_expand=None,
                     layers_not_to_expand=None,
                     add_layer_to_input_data=None,
                     remove_layers_from_input_data=None,
                     slice_length=None,
                     cast_data_type=None,
                     rename_inputs=None,
                     grayscale=None,
                     convert_to_torch=None,
                     names_to_indices=None,
                     assign_indices=None,
                     **kwargs):
    """
    Construct data preprocessing pipeline given basic values.

    :param batch:   Data batch
    :param h:   Data height
    :param w:   Data width
    :param resize_mode:   Interpolation
    :param mean:   Mean tuple (for 3 channels: x, y, z) to subtract from data.
                   Mutually exclusive with `mean_file`
    :param mean_file:   Mean file to subtract from data. Mutually exclusive
                        with `mean`
    :param grayscale:   Grayscale data
    :param cast_data_type:   Converts data type preprocessor
    :param scale_factor:   Data scaling factor
    :param normalization_factor:   Data normalization factor
    :param reverse_channels:   Apply reversing of channels (for images:
                                RGB<->BGR)
    :param permute_order:   Permute data shape. (i.e. from HWC:(0, 1, 2) to
                            CHW:(2, 0, 1) where H is height, W is width,
                            C is depth(channels))
    :param expand_dims:   Indicates whether to expand 0th dimension
    :param target_layers:   Target layers to apply preprocessing to
    :param slice_length:   Updates data through slice
    :param layers_to_expand:   Layers to apply preprocessor which allows expand dims
    :param layers_not_to_expand:   Layers to disable preprocessor which allows expand dims
    :param add_layer_to_input_data:   Add new input to input dictionary loaded from npz
    :param remove_layers_from_input_data:   Delete input from input dictionary loaded from npz
    :param rename_inputs:   Rename data layers of format: list of
                            (old name, new name)
    :param convert_to_torch: convert inputs to torch.Tensor format
    :param names_to_indices: whether convert input names to indices
    :param assign_indices: whether assign indices as input names for tensors
    :return:   "preprocess" step parsed by testing framework
    """

    def step_include(value):
        if not value:
            return False
        else:
            return True

    steps = []

    if step_include(assign_indices):
        steps.append(("assign_indices", {
            "target_layers": target_layers
        }))

    if step_include(remove_layers_from_input_data):
        steps.append(("remove_layers_from_input_data", {
            "target_layers": remove_layers_from_input_data
        }))

    if h and w:
        steps.append(("resize", {
            "height": h,
            "width": w,
            "mode=": resize_mode,
            "target_layers": target_layers
        }))

    if step_include(mean) and step_include(mean_file):
        raise AttributeError('both mean and mean file options specified')
    elif step_include(mean):
        steps.append(("subtract_mean_values", {
            "mean_values": mean,
            "target_layers": target_layers
        }))
    elif step_include(mean_file):
        steps.append(("subtract_mean_values_file", {
            "mean_file": mean_file,
            "target_layers": target_layers
        }))

    if grayscale == dict():
        steps.append(("grayscale", grayscale))

    if step_include(scale_factor) and step_include(normalization_factor):
        raise AttributeError(
            'both scale and normalization factors are specified')
    elif step_include(normalization_factor):
        steps.append(("normalize", {
            "factor": normalization_factor,
            "target_layers": target_layers
        }))
    elif step_include(scale_factor):
        steps.append(("scale", {
            "factor": scale_factor,
            "target_layers": target_layers
        }))

    if step_include(cast_data_type):
        steps.append(("cast_data_type", {"target_data_type": cast_data_type}))

    if step_include(slice_length):
        steps.append(("slice_data", {"slice": slice(None, slice_length, None)}))

    if step_include(reverse_channels):
        steps.append(("reverse_channels", {"target_layers": target_layers}))

    if step_include(permute_order):
        steps.append(("permute_shape", {
            "order": permute_order,
            "target_layers": target_layers
        }))

    if step_include(batch):
        steps.append(("align_with_batch", {
            "batch": batch,
            "expand_dims": expand_dims,
            "target_layers": target_layers
        }))

    if step_include(layers_to_expand) and step_include(layers_not_to_expand):
        steps.append(("align_with_batch_dif", {
            "batch": batch,
            "layers_to_expand": layers_to_expand,
            "layers_not_to_expand": layers_not_to_expand
        }))

    if step_include(add_layer_to_input_data):
        steps.append(("add_layer_to_input_data", {
            "layer_data": add_layer_to_input_data
        }))

    for preproc, config in kwargs.items():
        steps.append((preproc, config))

    # must be the last step
    # otherwise, preprocessors that depend on target_layers might fail
    if step_include(rename_inputs):
        steps.append(("rename_inputs", {"rename_input_pairs": rename_inputs}))

    if step_include(convert_to_torch):
        steps.append(("convert_to_torch", {
            "target_layers": target_layers
        }))

    if step_include(names_to_indices):
        steps.append(("names_to_indices", {
            "target_layers": target_layers
        }))

    return 'preprocess', OrderedDict(steps)


def assemble_preproc_tf(*args, **kwargs):
    """Add tensorflow-specific preprocessings. Pass rest of arguments as is.

    :return: "preprocess" step with TensorFlow specific actions
    """

    order = kwargs.pop('permute_order', None)
    reverse = kwargs.pop('reverse_channels', True)

    return assemble_preproc(*args, permute_order=order, reverse_channels=reverse, **kwargs)

