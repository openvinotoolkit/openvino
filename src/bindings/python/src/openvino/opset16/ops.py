# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset16."""
from functools import partial
from typing import Optional

from openvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import NodeInput, as_nodes

_get_node_factory_opset16 = partial(_get_node_factory, "opset16")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def identity(
    data: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Identity operation is used as a placeholder. It creates a copy of the input to forward to the output.

    :param data: Tensor with data.

    :return: The new node performing Identity operation.
    """
    return _get_node_factory_opset16().create(
        "Identity",
        as_nodes(data, name=name),
        {},
    )


@nameable_op
def istft(
    data: NodeInput,
    window: NodeInput,
    frame_size: NodeInput,
    frame_step: NodeInput,
    center: bool,
    normalized: bool,
    signal_length: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which generates ISTFT operation.

    :param  data: The node providing input data.
    :param  window: The node providing window data.
    :param  frame_size: The node with scalar value representing the size of Fourier Transform.
    :param  frame_step: The distance (number of samples) between successive window frames.
    :param  center: Flag signaling if the signal input has been padded before STFT.
    :param  normalized: Flag signaling if the STFT result has been normalized.
    :param  signal_length: The optional node with length of the original signal.
    :param  name: The optional name for the created output node.
    :return: The new node performing ISTFT operation.
    """
    if signal_length is None:
        inputs = as_nodes(data, window, frame_size, frame_step, name=name)
    else:
        inputs = as_nodes(data, window, frame_size, frame_step, signal_length, name=name)
    return _get_node_factory_opset16().create(
        "ISTFT",
        inputs,
        {"center": center, "normalized": normalized},
    )
