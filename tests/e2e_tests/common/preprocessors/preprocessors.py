# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data preprocessors applied to target layers in given data dictionary."""
import logging as log
import sys

# pylint:disable=no-member
import cv2
import numpy as np

from e2e_tests.test_utils.path_utils import resolve_file_path
from .provider import ClassProvider

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


class Squeeze(ClassProvider):
    """Squeezing preprocessor.

    Removes single-dimensional entries from the shape of an array.
    """
    __action_name__ = "squeeze"

    def __init__(self, config):
        self.axis = config.get('axis', None)
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Apply np.squeeze to data."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        for layer in apply_to:
            if self.axis is not None:
                # If an axis is selected with shape entry greater than one, an error is raised
                assert np.all(np.array(data[layer].shape).take(self.axis) == 1), \
                    'Squeeze preprocessor error: Can not squeeze data for layer {} with shape {} by axis {}' \
                    ''.format(layer, data[layer].shape, self.axis)
            data[layer] = np.squeeze(data[layer], axis=self.axis)
        return data


class Resize(ClassProvider):
    """Resize preprocessor.

    Resizes data to HEIGHTxWIDTH using optional interpolation MODE.
    """
    __action_name__ = "resize"
    resize_interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    def __init__(self, config):
        self.height = int(config["height"])
        self.width = int(config["width"])
        self.target_layers = config.get('target_layers', None)
        self.interpolation = Resize.resize_interp_map[config.get(
            'mode', 'linear')]

    def apply(self, data):
        """Resize data with opencv resize."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info(
            "Resize input data for layers {} to size ({}, {}) ...".format(', '.join('"{}"'.format(l) for l in apply_to),
                                                                          self.width,
                                                                          self.height))
        for layer in apply_to:
            data[layer] = cv2.resize(data[layer], (self.width, self.height), interpolation=self.interpolation)
        return data


class PermuteShape(ClassProvider):
    """Shape permutation preprocessor.

    Permutes data shape using ORDER value.
    """
    __action_name__ = "permute_shape"

    def __init__(self, config):
        self.order = config["order"]
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Apply np.transpose to data."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info(
            "Permute input data for layers {} to order {} ...".format(', '.join('"{}"'.format(l) for l in apply_to),
                                                                      self.order))
        for layer in apply_to:
            data[layer] = np.transpose(data[layer], self.order)
        return data


class AlignWithBatch(ClassProvider):
    """Batch alignment preprocessor.

    Aligns batch dimension in input data
    with BATCH value specified in test.

    Models 0-th dimension for batch and
    duplicates input data while size of batch
    dimension in input data won't be equal with BATCH.
    """
    __action_name__ = "align_with_batch"

    def __init__(self, config):
        self.batch = config["batch"]
        self.batch_dim = config.get('batch_dim', 0)
        self.expand_dims = config.get('expand_dims', True)
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Apply batch alignment (duplication) to data."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Align batch data for layers {} to batch {} ...".format(', '.join(
            '"{}"'.format(l) for l in apply_to), self.batch))
        for layer in apply_to:
            if self.expand_dims:
                data[layer] = np.expand_dims(data[layer], axis=self.batch_dim)
            data[layer] = np.repeat(data[layer], self.batch, axis=self.batch_dim)
        return data


class SubtractMeanValues(ClassProvider):
    """Mean subtraction preprocessor."""
    __action_name__ = "subtract_mean_values"

    def __init__(self, config):
        self.mean_values = config["mean_values"]
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Subtract mean values from data."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Subtract mean values {} from input data for layers {} ...".format(self.mean_values, ', '.join(
            '"{}"'.format(l) for l in apply_to)))
        for layer in apply_to:
            data[layer] = data[layer] - self.mean_values
        return data


class SubtractMeanValuesFile(ClassProvider):
    """Mean file (image) subtraction preprocessor."""
    __action_name__ = "subtract_mean_values_file"

    def __init__(self, config):
        self.mean_file = resolve_file_path(config['mean_file'], as_str=True)
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Subtract mean image from data."""
        means = None
        with np.load(self.mean_file) as content:
            means = content['means']
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info(
            "Subtract mean file {} from input data for layers {} ...".format(self.mean_file, ', '.join(
                '"{}"'.format(l) for l in apply_to)))
        for layer in apply_to:
            data_shape = data[layer].shape
            if len(data_shape) != 3:
                raise ValueError('data layer {l} has unexpected shape {s}, '
                                 'expected shape of length 3.'.format(l=layer, s=data_shape))
            mean_values = means[:data_shape[0], :data_shape[1], :]
            data[layer] = data[layer] - mean_values
        return data


class Normalize(ClassProvider):
    """Normalization preprocessor."""
    __action_name__ = "normalize"

    def __init__(self, config):
        self.factor = config["factor"]
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Normalize data by factor."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Normalize input data for layers {} with normalization factor {}...".format(
            ', '.join('"{}"'.format(l) for l in apply_to),
            self.factor))
        for layer in apply_to:
            data[layer] = data[layer] / self.factor
        return data


class ExpandDims(ClassProvider):
    """Expand dims preprocessor."""
    __action_name__ = "expand_dims"

    def __init__(self, config):
        self.target_layers = config.get('target_layers', None)
        self.axis = config.get('axis', 0)

    def apply(self, data):
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Expanding dims for layers {}...".format(', '.join('"{}"'.format(l) for l in apply_to)))
        for layer in apply_to:
            data[layer] = np.expand_dims(data[layer], axis=self.axis)
        return data


class Scale(ClassProvider):
    """Scale preprocessor."""
    __action_name__ = "scale"

    def __init__(self, config):
        self.factor = config["factor"]
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Scale data by factor."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Scale input data for layers {} with scaling factor {} ...".
                 format(', '.join('"{}"'.format(l) for l in apply_to), self.factor))
        for layer in apply_to:
            data[layer] = data[layer] * self.factor
        return data


class ReverseChannels(ClassProvider):
    """Channel reverse preprocessor.

    Reverses channels in data (i.e. RGB image -> BGR image).
    """
    __action_name__ = "reverse_channels"

    def __init__(self, config):
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Apply cv2.cvtColor to data."""

        def convert(data):
            """OpenCV color conversion"""
            # cvtColor doesn't seem to be supported for float64
            # converting to float32 first, then applying color convert
            # return in original type
            orig_type = data.dtype
            return cv2.cvtColor(data.astype(np.float32), cv2.COLOR_RGB2BGR).astype(orig_type)

        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Convert colors from RGB to BGR for input data for layers {} ...".format(
            ', '.join('"{}"'.format(l) for l in apply_to)))
        for layer in apply_to:
            if len(data[layer].shape) != 3:
                raise ValueError(
                    'data layer {l} has unexpected shape {s}, ''expected shape of length 3.'.format(l=layer, s=data[
                        layer].shape))
            data[layer] = convert(data[layer])
        return data


class RenameInputs(ClassProvider):
    """Input renaming preprocessor."""
    __action_name__ = "rename_inputs"

    def __init__(self, config):
        self.input_pairs = config.get('rename_input_pairs', [])

    def apply(self, data):
        """Rename data keys."""
        if self.input_pairs:
            log.info("Rename input data keys according to pairs {}...".format(self.input_pairs))
        for old_name, new_name in self.input_pairs:
            data[new_name] = data.pop(old_name)
        return data


class RemoveLayersFromInputData(ClassProvider):
    """Updating input data preprocessor.

    Removes input layers from input data
    Use case: if some input layer freezed with value during convertion model
    in IR, need to remove this layer from input data read from file
    """
    __action_name__ = "remove_layers_from_input_data"

    def __init__(self, config):
        self.target_layers = config.get('target_layers', [])

    def apply(self, data):
        """Remove layers from input data."""
        for layer in self.target_layers:
            data.pop(layer)
        return data


class AddLayersToInputData(ClassProvider):
    """Updating input data preprocessor.

    Add input layers to input data
    Use case: if some input layer depends on height or weight of previous input layer,
    It is need to dynamically fill this layer with value and not to hardcode.
    """
    __action_name__ = "add_layer_to_input_data"

    def __init__(self, config):
        self.layer_data = config["layer_data"]

    def apply(self, data):
        """Add layer to input data."""
        for key in self.layer_data.keys():
            log.info("Adding layer to input data: layer {}...".format(key))
            data.update({key: np.array(self.layer_data[key])})
        return data


class CopyDataFromLayer(ClassProvider):
    """Copying data from one layer to another"""
    __action_name__ = "copy_data_from_layer"

    def __init__(self, config):
        self.source_target_map = config.get("source_target_map", {})

    def apply(self, data):
        """Apply rewrite of sequence_length value in input data."""
        for source, target in self.source_target_map.items():
            data = AddLayersToInputData({"layer_data": {target: data[source]}}).apply(data)
        return data


class RewriteSeqLenValue(ClassProvider):
    """Sequence_length rewriting preprocessor.

    Changes sequence_length value in input_data on SEQUENCE_LENGTH value from the test.
    """
    __action_name__ = "rewrite_seqlen_value"

    def __init__(self, config):
        self.sequence_length = config["sequence_length"]
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Apply rewrite of sequence_length value in input data."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        for layer in apply_to:
            data[layer] = np.array([self.sequence_length])
        return data


class SliceData(ClassProvider):
    """Slice preprocessor.

    Updates input data through slice
    Use case: align size of dimension with value specified
    in test (e.g. align with batch)

    Config should have 'slice' field with 'slice' or
    'tuple of slices' types of value.

    Examples how to model some types of slices:
        slice(0, 5, 1) = slice(0, 5, 1)
        [:1, 3:5:2] = (slice(None, 1, None), slice(3, 5, 2))
    """

    __action_name__ = "slice_data"

    def __init__(self, config):
        self.slice = config["slice"]
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        for layer in apply_to:
            data[layer] = data[layer][self.slice]
        return data


class CastDataType(ClassProvider):
    """Converts data type preprocessor"""
    __action_name__ = "cast_data_type"

    def __init__(self, config):
        self.target_data_type = config.get('target_data_type', "float32")
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Converts data type to specified 'target_data_type' for provided numpy.ndarray"""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Converting layers {} data to type {}...".format(', '.join('"{}"'.format(l) for l in apply_to),
                                                                  self.target_data_type))
        for layer in apply_to:
            data[layer] = data[layer].astype(self.target_data_type)
        return data


class CustomPreproc(ClassProvider):
    __action_name__ = "custom_preproc"

    def __init__(self, config):
        self.execution_function = config["execution_function"]

    def apply(self, data):
        return self.execution_function(data)


class DynamismPreproc(CustomPreproc):
    """Implementation duplicates CustomPreproc preprocessor."""
    __action_name__ = "dynamism_preproc"

    pass


class Grayscale(ClassProvider):
    """Convert image to grayscale preprocessor."""
    __action_name__ = "grayscale"

    def __init__(self, config):
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Scale data by factor."""
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Converting layers {} to grayscale ...".format(', '.join('"{}"'.format(l) for l in apply_to)))
        for layer in apply_to:
            data[layer] = cv2.cvtColor(data[layer], cv2.COLOR_BGR2GRAY)
            data[layer] = np.expand_dims(data[layer], axis=2)
        return data


class AlignWithBatchDifferently(ClassProvider):
    """Align with batch preprocessor which allows expand dims for chosen layers"""
    # TODO: Replace with more accurate solution
    __action_name__ = "align_with_batch_dif"

    def __init__(self, config):
        self.batch = config["batch"]
        self.batch_dim = config.get('batch_dim', 0)
        self.expand_dims = config.get('expand_dims', True)
        self.layers_to_expand = config.get('layers_to_expand', None)
        self.layers_not_to_expand = config.get('layers_not_to_expand', None)

    def apply(self, data):
        if self.layers_to_expand:
            data_preproc = AlignWithBatch({"batch": self.batch, "target_layers": self.layers_to_expand})
            data = data_preproc.apply(data)
        else:
            log.warning("No layers specified to be aligned with batch using dimension expanding.")
        if self.layers_not_to_expand:
            data_preproc = AlignWithBatch(
                {"batch": self.batch, "expand_dims": False, "target_layers": self.layers_not_to_expand})
            data = data_preproc.apply(data)
        else:
            log.warning("No layers specified to be aligned with batch using no dimension expanding.")
        return data


class ConvertToTorchTensor(ClassProvider):
    """Convert arrays to torch.Tensor format."""
    __action_name__ = "convert_to_torch"

    def __init__(self, config):
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        import torch
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Converting layers {} to torch.Tensor format ...".format(', '.join('"{}"'.format(l) for l in apply_to)))
        for layer in apply_to:
            data[layer] = torch.from_numpy(data[layer])
        return data


class ConvertNamesToIndices(ClassProvider):
    """Converts input names to indices"""
    __action_name__ = "names_to_indices"

    def __init__(self, config):
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        converted = {}
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
        log.info("Converting names {} to indices ...".format(', '.join('"{}"'.format(l) for l in apply_to)))
        for i, layer in enumerate(apply_to):
            converted[i] = data[layer]
        return converted


class AssignIndices(ClassProvider):
    """Assigns indices for tensors"""
    __action_name__ = "assign_indices"

    def __init__(self, config):
        pass

    @staticmethod
    def apply(data):
        import torch
        if isinstance(data, (torch.Tensor, np.ndarray)):
            data = [data]
        converted = {}
        for i in range(len(data)):
            converted[i] = data[i]
        return converted

