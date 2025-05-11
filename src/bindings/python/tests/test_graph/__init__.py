# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# openvino.dll directory path visibility is needed to use _pyopenvino module
# import below causes adding this path to os.environ["PATH"]
import openvino  # noqa: F401 'imported but unused'
