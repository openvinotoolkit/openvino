# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""IE network modifiers applied to IE network."""

import logging as log
import sys

from e2e_tests.utils.test_utils import align_input_names
from e2e_tests.common.test_utils import get_shapes_from_data, convert_shapes_to_partial_shape
from .container import ClassProvider

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


class ReshapeInputShape(ClassProvider):
    """Reshape IE network modifier.

    Reshapes IE network on the same shapes of
    the corresponding input data.
    """

    __action_name__ = "reshape_input_shape"

    def __init__(self, config):
        self.path = config["input_path"]

    def apply(self, network, input_data):
        shapes = get_shapes_from_data(input_data, api_version='2')
        log.info("OV Model will be reshaped on {}".format(shapes))
        network.reshape(shapes)
        return network


class ReshapeCurrentShape(ClassProvider):
    """Reshape IE network modifier.

    Reshapes IE network on the same shapes of
    the corresponding input layers
    """

    __action_name__ = "reshape_current_shape"

    def __init__(self, config):
        pass

    def apply(self, network, **kwargs):
        shapes = {}
        for input in network.input_info:
            shapes[input] = network.input_info[input].input_data.shape
        log.info("IE Network will be reshaped on {}".format(shapes))
        network.reshape(shapes)
        return network


class Reshape(ClassProvider):
    """Reshape IE network modifier.

    Reshapes IE network on shapes specified in config

    Config should have 'shapes' field with dictionary
    where keys are input layers' names and values are
    corresponding input shapes.

    Example:
        shapes = {"Placeholder": (1, 224, 224, 3)}
    """

    __action_name__ = "reshape"

    def __init__(self, config):
        self.shapes = config["shapes"]

    def apply(self, network, **kwargs):
        log.info("OV Model will be reshaped on {}".format(self.shapes))
        self.shapes = convert_shapes_to_partial_shape(self.shapes)
        network.reshape(align_input_names(self.shapes, network))
        return network


class SetBatchReshape(ClassProvider):
    """Batch IE network modifier.

    Sets batch of IE network to BATCH value specified in config

    Config should have 'batch' field with 'int' value.
    """

    __action_name__ = "set_batch_using_reshape"

    def __init__(self, config):
        self.batch = config["batch"]
        self.target_layers = config.get('target_layers', None)
        self.batch_dim = config.get('batch_dim', 0)

    def apply(self, network, **kwargs):
        log.info("OV Model's batch will be set to {}".format(self.batch))
        input_shapes = {}
        for network_input in network.inputs:
            input_name = network_input.get_any_name()
            if self.target_layers and input_name not in self.target_layers:
                common_names = network_input.names.intersection(set(self.target_layers))
                if common_names:
                    input_name = common_names.pop()
            input_shapes[input_name] = network_input.get_partial_shape()

        apply_to = self.target_layers if self.target_layers is not None else input_shapes.keys()

        reshaped = False
        for layer in apply_to:
            if input_shapes[layer][self.batch_dim] == self.batch:
                log.info("For layer '{}' target shape {} "
                         "equals to initial shape, no reshape done".format(layer, input_shapes[layer]))
                continue
            input_shapes[layer][self.batch_dim] = self.batch
            reshaped = True
        if reshaped:
            network.reshape(input_shapes)
        return network


class AddOutputs(ClassProvider):
    """Network outputs modifier.

    Adds additional outputs to the network allowing to get intermediate tensors.

    Config should have 'outputs' field with tuples ("node_name", output_port) or a single element "node_name". In the
    latter case the output port is implicitly set to 0.
    """

    __action_name__ = "add_outputs"

    def __init__(self, config):
        self.outputs = config["outputs"]
        assert self.outputs is not None, 'The "outputs" must be specified for the Network output modifier'

    def apply(self, network, **kwargs):
        log.info("IE Network outputs will be expanded with the following ones: {}".format(self.outputs))
        network.add_outputs(self.outputs)
        return network
