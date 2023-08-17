# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.utils import _add_openvino_libs_to_search_path

_add_openvino_libs_to_search_path()

from openvino.frontend.pytorch.torchdynamo import backend
