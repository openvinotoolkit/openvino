# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from openvino import Core, Extension
from openvino.op import _GroupQueryAttentionExtension


def test_gqa_extension_is_extension():
    ext = _GroupQueryAttentionExtension()
    assert isinstance(ext, Extension)


def test_gqa_extension_add_to_core():
    ext = _GroupQueryAttentionExtension()
    core = Core()
    core.add_extension(ext)  # must not raise
