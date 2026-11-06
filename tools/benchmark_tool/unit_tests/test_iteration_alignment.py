# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino.tools.benchmark.utils.utils import get_number_iterations


@pytest.mark.parametrize('iterations,requests,shapes,expected', [
    (6, 2, 3, 6), (7, 2, 3, 9), (1, 2, 3, 3), (2, 2, 3, 3), (100, 2, 3, 102),
    (5, 3, 1, 6), (6, 3, 1, 6), (0, 2, 3, 0),
    (2**54 + 1, 1, 3, ((2**54 + 3) // 3) * 3),
])
def test_async_iterations_round_up_to_complete_groups(iterations, requests, shapes, expected):
    assert get_number_iterations(iterations, requests, shapes, 'async') == expected


def test_sync_iterations_remain_unchanged():
    assert get_number_iterations(7, 2, 3, 'sync') == 7
