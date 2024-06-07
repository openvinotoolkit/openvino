# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# type: ignore
# flake8: noqa

from openvino._pyopenvino.passes import ModelPass, Matcher, MatcherPass, PassBase, WrapType, Or, AnyInput, Optional
from openvino._pyopenvino.passes import (
    consumers_count,
    has_static_dim,
    has_static_dims,
    has_static_shape,
    has_static_rank,
    rank_equals,
    type_matches,
    type_matches_any,
)
from openvino._pyopenvino.passes import Serialize, ConstantFolding, VisualizeTree, MakeStateful, LowLatency2, ConvertFP32ToFP16, Version
from openvino.runtime.passes.manager import Manager
from openvino.runtime.passes.graph_rewrite import GraphRewrite, BackwardGraphRewrite
