# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from openvino.frontend.pytorch.torchdynamo.backend_utils import _is_testing


@pytest.mark.parametrize(
    ("testing_value", "expected"),
    [
        ("false", False),
        ("0", False),
        (False, False),
        ("true", True),
        ("1", True),
        (True, True),
    ],
)
def test_is_testing_flag(testing_value, expected):
    assert _is_testing({"testing": testing_value}) == expected