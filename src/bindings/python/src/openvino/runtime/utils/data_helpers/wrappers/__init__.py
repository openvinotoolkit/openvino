# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino._pyopenvino import Tensor, ConstOutput
from openvino._pyopenvino import InferRequest as InferRequestBase

from openvino.utils.data_helpers.wrappers import tensor_from_file
from openvino.utils.data_helpers.wrappers import _InferRequestWrapper
from openvino.utils.data_helpers.wrappers import OVDict
