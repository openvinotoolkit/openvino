
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import typing

class ModuleExtension:
    def __init__(self, module, extension=None, replacer=None):
        self.module = module
        self.extension = extension
        self.replacer = replacer