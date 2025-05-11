# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""File readers."""
# pylint:disable=no-member
import numpy as np
import cv2
import logging as log
import sys
from copy import deepcopy

from e2e_tests.test_utils.path_utils import resolve_file_path
from e2e_tests.test_utils.tf_hub_utils import prepare_inputs, get_inputs_info
from e2e_tests.common.readers.provider import ClassProvider

try:
    from onnx import TensorProto, numpy_helper

    onnx_not_installed = False
except ImportError:
    onnx_not_installed = True


class NPZReader(ClassProvider):
    """
    Read input data from .npz file - most preferable input reading method.
    Config should have 'path' field (absolute or relative to 'input_data'
    defined in env_config.yml). File should store zipped dictionary of input
    layer names as keys, and appropriate input data.
    """
    __action_name__ = "npz"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        """Initialization method.

        :param config: configuration dict. must have 'path' key
        """
        self.input_path = resolve_file_path(config['path'], as_str=True)

    def read(self):
        """Return file content."""
        log.info("Reading input file from {} ...".format(self.input_path))
        return dict(np.load(self.input_path, allow_pickle=True))


class NPYReader(ClassProvider):
    """
    Read input data from .npy file. Config should have 'path' field with mapping
    (dictionary) of input layer name and corresponding .npy file.
    """
    __action_name__ = "npy"

    def __init__(self, config):
        """Initialization method.
        :param config: configuration dict. must have 'path' key
        """
        self.inputs_map = config['inputs_map']

    def read(self):
        """Return file content."""
        input_data = {}
        for input, path in self.inputs_map.items():
            log.info("Reading input file from {} for input '{}' ...".format(path, input))
            input_data[input] = np.load(path, allow_pickle=True)
        return input_data


class ImageReader(ClassProvider):
    """
    Read input data from image file. Config should have 'inputs_map' field with mapping
    (dictionary) of input layer name and corresponding image path.
    """
    __action_name__ = "img"

    def __init__(self, config):
        """Initialization method.
        :param config: configuration dict. must have 'path' key
        """
        self.inputs_map = config['inputs_map']

    def read(self):
        """Return image data."""
        input_data = {}
        for input, path in self.inputs_map.items():
            log.info("Reading input file from {} for input '{}' ...".format(path, input))
            input_data[input] = cv2.imread(path)
        return input_data


class ProtobufReader(ClassProvider):
    """
    Read input data in protobuf format. Config should have 'path' field (absolute or relative to 'input_data'
    defined in env_config.yml). File should store data encoded with protobuf."""
    __action_name__ = "pb"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        """Initialization method.
        :param config: configuration dict. must have 'path' key
        """
        self.inputs_map = config['inputs_map']

    def read(self):
        if onnx_not_installed:
            raise RuntimeError("ONNX module not available")
        input_data = {}
        tensor = TensorProto()
        for input_name, input_path in self.inputs_map.items():
            log.info("Reading input file for input {} from {} ...".format(input_name, input_path))
            with open(input_path, 'rb') as f:
                tensor.ParseFromString(f.read())
                input_data[input_name] = numpy_helper.to_array(tensor)
        return input_data


class ExternalData(ClassProvider):
    __action_name__ = "external_data"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        self.data = deepcopy(config['data'])
        assert isinstance(self.data, (dict, list)), \
            "External input data specified in config key 'data' have to be a " \
            "dictionary or list of dictionaries with input layer names as keys and numpy.ndarrays with " \
            "input data as values"

    def read(self):
        return self.data


class TorchReader(ClassProvider):
    """
    Read input data from .pt or .pth file.
    All content in file should be stored in list.
    Config should have 'path' field (absolute or relative to 'input_data'
    defined in env_config.yml).
    """
    __action_name__ = "pt"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config):
        """Initialization method.

        :param config: configuration dict. must have 'path' key
        """
        self.input_path = resolve_file_path(config['path'], as_str=True)

    def read(self):
        """Return file content."""
        import torch
        log.info("Reading input file from {} ...".format(self.input_path))
        return torch.load(self.input_path)


class TFHubInputsGenerator(ClassProvider):
    """
    Generates random inputs depending on model's input type
    """
    __action_name__ = "generate_tf_hub_inputs"
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    def __init__(self, config=None):
        """Initialization method.
        """
        self.config = config

    def read(self, tf_hub_model):
        """Return file content."""
        return prepare_inputs(get_inputs_info(tf_hub_model))
