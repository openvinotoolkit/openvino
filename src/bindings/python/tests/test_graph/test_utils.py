# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino._pyopenvino.util import deprecation_warning


def test_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="function1 is deprecated"):
        deprecation_warning("function1")
    with pytest.warns(DeprecationWarning, match="function2 is deprecated and will be removed in version 2025.4"):
        deprecation_warning("function2", "2025.4")
    with pytest.warns(DeprecationWarning, match="function3 is deprecated. Use another function instead"):
        deprecation_warning("function3", message="Use another function instead")
    with pytest.warns(DeprecationWarning, match="function4 is deprecated and will be removed in version 2025.4. Use another function instead"):
        deprecation_warning("function4", version="2025.4", message="Use another function instead")
