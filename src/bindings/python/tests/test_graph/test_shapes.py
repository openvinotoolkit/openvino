# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino import Shape, PartialShape

@pytest.mark.parametrize("cls", [Shape, PartialShape])
def test_single_index(cls):
    shape = cls([1, 2, 3, 4, 5])
    assert shape[0] == 1
    assert shape[2] == 3

@pytest.mark.parametrize("cls", [Shape, PartialShape])
def test_negative_single_index(cls):
    shape = cls([1, 2, 3, 4, 5])
    assert shape[-1] == 5
    assert shape[-3] == 3

@pytest.mark.parametrize("cls", [Shape, PartialShape])
def test_slicing_step(cls):
    shape = cls([1, 2, 3, 4, 5])
    assert list(shape[::2]) == [1, 3, 5]
    assert list(shape[1::2]) == [2, 4]
