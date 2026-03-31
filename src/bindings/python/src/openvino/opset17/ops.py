# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset17."""
from functools import partial

from openvino.utils.node_factory import _get_node_factory

_get_node_factory_opset17 = partial(_get_node_factory, "opset17")

# -------------------------------------------- ops ------------------------------------------------
