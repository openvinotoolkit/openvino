// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class TransposeConv1x1TransposeFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("TransposeConv1x1TransposeFusion");
    TransposeConv1x1TransposeFusion(bool supports_immad = false);
};

class TransposeConv1x1TransposeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeConv1x1TransposeMatcher");
    TransposeConv1x1TransposeMatcher(bool supports_immad);
};

}  // namespace ov::intel_gpu