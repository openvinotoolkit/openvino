# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest


skip_devtest = pytest.mark.skip(reason="Test might depend on machine, should be run by developers"
                                       "or advanced users for debug/testing purposes.")
skip_need_mock_op = pytest.mark.skip(reason="Test need to be rewritten with mock operation. Issue #101215")
