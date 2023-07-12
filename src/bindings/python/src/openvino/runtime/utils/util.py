# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino._pyopenvino.util import clone_model as clone_model_base

# mypy: allow-untyped-defs


def clone_model(model):
    from openvino.runtime import Model
    return Model(clone_model_base(model))
