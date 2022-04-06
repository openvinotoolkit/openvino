# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from openvino.tools.pot.graph import node_utils as nu
from openvino.tools.pot.utils.logger import get_logger


logger = get_logger(__name__)


def get_optimization_params(loss_name, optimizer_name):
    loss_fn_map = {
        'l2': torch.nn.MSELoss(),
    }

    optimizer_map = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
    }
    return loss_fn_map[loss_name], optimizer_map[optimizer_name]


def get_weight_node(node, port_id=1):
    node_weight = nu.get_node_input(node, port_id)
    if node_weight.type == 'FakeQuantize':
        node_weight = nu.get_node_input(node_weight, 0)
    if node_weight.type != 'Const':
        raise ValueError('Provided weight node is not Const!')
    return node_weight
