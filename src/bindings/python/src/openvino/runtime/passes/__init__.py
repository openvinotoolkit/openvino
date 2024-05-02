# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# type: ignore
# flake8: noqa

from openvino._pyopenvino.passes import (
    AnyInput,
    ConstantFolding,
    ConvertFP32ToFP16,
    LowLatency2,
    MakeStateful,
    Matcher,
    MatcherPass,
    ModelPass,
    Optional,
    Or,
    PassBase,
    Serialize,
    Version,
    VisualizeTree,
    WrapType,
    consumers_count,
    has_static_dim,
    has_static_dims,
    has_static_rank,
    has_static_shape,
    rank_equals,
    type_matches,
    type_matches_any,
)
from openvino.runtime.passes.graph_rewrite import BackwardGraphRewrite, GraphRewrite
from openvino.runtime.passes.manager import Manager
