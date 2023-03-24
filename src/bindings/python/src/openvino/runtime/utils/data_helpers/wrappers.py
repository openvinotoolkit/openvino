# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino._pyopenvino import Tensor
from openvino._pyopenvino import InferRequest as InferRequestBase


def tensor_from_file(path: str) -> Tensor:
    """Create Tensor from file. Data will be read with dtype of unit8."""
    return Tensor(np.fromfile(path, dtype=np.uint8))  # type: ignore


class _InferRequestWrapper(InferRequestBase):
    """InferRequest class with internal memory."""

    def __init__(self, other: InferRequestBase) -> None:
        # Private memeber to store newly created shared memory data
        self._inputs_data = None
        super().__init__(other)
