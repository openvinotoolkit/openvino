# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from openvino import PartialShape
from openvino.runtime import opset13 as ops
from openvino.runtime.passes import Matcher, WrapType, Or, AnyInput, Optional
from openvino.runtime.passes import (
    consumers_count,
    has_static_dim,
    has_static_dims,
    has_static_shape,
    has_static_rank,
    type_matches,
    type_matches_any,
)

# ! [ov:create_simple_model]
def simple_model():
    pass

# ! [ov:create_simple_model]