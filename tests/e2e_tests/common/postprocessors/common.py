# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Common postprocessors."""
import numpy as np

from e2e_tests.common.preprocessors.preprocessors import SliceData, Normalize, CustomPreproc, RemoveLayersFromInputData, \
    RenameInputs, Squeeze
from .provider import ClassProvider


class Squeeze(ClassProvider, Squeeze):
    """Squeezing postprocessor.

    Implementation duplicates Squeeze preprocessor.
    """
    __action_name__ = "squeeze"
    pass


class AlignWithBatch(ClassProvider):
    """Batch alignment postprocessor.

    Duplicates 1-batch data BATCH number of times.
    """
    __action_name__ = "align_with_batch"

    def __init__(self, config):
        self.batch = config['batch']
        self.batch_dim = config.get('batch_dim', 0)
        self.expand_dims = config.get('expand_dims', False)
        self.target_layers = config.get('target_layers', None)
        self.axis = config.get('axis', [self.batch_dim])

    def apply(self, data):
        """Apply batch alignment (duplication) to data."""
        apply_to = self.target_layers if self.target_layers else data.keys()
        for layer in apply_to:
            if self.expand_dims:
                data[layer] = np.expand_dims(data[layer], axis=self.batch_dim)
            for axis in self.axis:
                data[layer] = np.repeat(data[layer], self.batch, axis=axis)
        return data


class FilterTorchData(ClassProvider):
    """Batch alignment postprocessor.

    Filters torch outputs and converts them to numpy format.
    """
    __action_name__ = "filter_torch_data"

    def __init__(self, config):
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        apply_to = self.target_layers if self.target_layers else data.keys()
        filtered_data = {}
        for layer in apply_to:
            filtered_data[layer] = data[layer].detach().numpy()
        return filtered_data


class PermuteShape(ClassProvider):
    """Shape permutation postprocessor.

    Permutes data shape using ORDER value.
    """
    __action_name__ = "permute_shape"

    def __init__(self, config):
        self.order = config["order"]
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        """Apply np.transpose to data."""
        apply_to = self.target_layers if self.target_layers else data.keys()
        for layer in apply_to:
            data[layer] = np.transpose(data[layer], self.order)
        return data


class RemoveLayer(ClassProvider):
    """Layer removal postprocessor.

    Removes layer from data dictionary by name.
    """
    __action_name__ = "remove_layer"

    def __init__(self, config):
        self.target_layers = config.get('layers_to_remove', None)

    def apply(self, data):
        if self.target_layers:
            for layer in self.target_layers:
                data.pop(layer)
        return data


class SliceData(ClassProvider, SliceData):
    """Slice postprocessor.

    Implementation duplicates SliceData preprocessor
    """

    pass


class Normalize(ClassProvider, Normalize):
    """Normalize postprocessor.

    Implementation duplicates Normalize preprocessor
    """
    pass


class ExpandDims(ClassProvider):
    """Dimension expanding postprocessor.

    Expands dimension by axis.
    """
    __action_name__ = "expand_dims"

    def __init__(self, config):
        self.target_layers = config.get('target_layers')
        self.axis = config.get('axis')

    def apply(self, data):
        self.target_layers = self.target_layers if self.target_layers else data.keys()
        for layer in self.target_layers:
            data[layer] = np.expand_dims(data[layer], axis=self.axis)
        return data


class RemoveZeros(ClassProvider):
    """Removing zeros postprocessor.

    Removes all-zero elements by axis.
    """
    __action_name__ = "remove_zeros"

    def __init__(self, config):
        self.target_layers = config.get('target_layers')
        self.axis = config.get('axis')

    def apply(self, data):
        self.target_layers = self.target_layers if self.target_layers else data.keys()
        for layer in self.target_layers:
            data[layer] = data[layer][np.any(data[layer], axis=self.axis)]
        return data


class Clip(ClassProvider):
    """Removing zeros postprocessor.

    Removes all-zero elements by axis.
    """
    __action_name__ = "clip"

    def __init__(self, config):
        self.target_layers = config.get('target_layers')
        self.min = config.get('min')
        self.max = config.get('max')

    def apply(self, data):
        self.target_layers = self.target_layers if self.target_layers else data.keys()
        for layer in self.target_layers:
            data[layer] = np.clip(data[layer], self.min, self.max)
        return data


class CustomPostproc(ClassProvider, CustomPreproc):
    """Custom postprocessor.

    Implementation duplicates CustomPreproc preprocessor
    """
    __action_name__ = "custom_postprocessor"

    pass


class RemoveLayersFromData(ClassProvider, RemoveLayersFromInputData):
    """Updating data postprocessor.

    Removes layers from data
    Use case: if reference results contain extra outputs for comparison,
    it can be removed from data through this postprocessor
    """
    __action_name__ = "remove_layers_from_data"

    pass


class RenameOutputs(ClassProvider, RenameInputs):
    __action_name__ = "rename_outputs"
    pass 


class ConvertNamesToIndices(ClassProvider):
    """Converts input names to indices"""
    __action_name__ = "names_to_indices"

    def __init__(self, config):
        self.target_layers = config.get('target_layers', None)

    def apply(self, data):
        converted = {}
        apply_to = self.target_layers if self.target_layers is not None else data.keys()
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
        if isinstance(data, torch.Tensor):
            data = [data]
        converted = {}
        for i in range(len(data)):
            converted[i] = data[i]
        return converted
