# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from openvino.tools.ovc.version import get_simplified_ie_version


@pytest.mark.parametrize("version, expected", [
    ("2025.4.0-20398-8fdad55727d-releases/2025/4", "2025.4"),
    ("2027.0.0-12345-abcdef1234", "2027.0"),
    ("2025.4.0-20398-8fdad55727d-custom_releases/2025/4", "2025.4"),
    ("2.1.2021.1.0-1028-55e4d5673a8", "2021.1"),
    ("2.1.custom_releases/2021/1_55e4d567", "2021.1"),
    ("custom_releases/2021/1_55e4d567", "2021.1"),
    ("custom_master_abc", "custom"),
])
def test_get_simplified_ie_version(version, expected):
    assert get_simplified_ie_version(version) == expected
