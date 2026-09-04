# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.frontend.pytorch.torchdynamo.backend_utils import _is_testing

def test_is_testing_flag_false():
    assert _is_testing({"testing": "false"}) == False
    assert _is_testing({"testing": "0"}) == False

def test_is_testing_flag_true():
    assert _is_testing({"testing": "true"}) == True
    assert _is_testing({"testing": "1"}) == True
    assert _is_testing({"testing": True}) == True