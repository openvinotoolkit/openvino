# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino._pyopenvino.util import clone_model as clone_model_base
from openvino.utils import deprecated

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openvino.runtime import Model


@deprecated(version="2024.0")
def clone_model(model: "Model") -> "Model":
    from openvino.runtime import Model
    return Model(clone_model_base(model))
