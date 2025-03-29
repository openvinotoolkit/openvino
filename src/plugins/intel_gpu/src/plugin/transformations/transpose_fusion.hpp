// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class TransposeFusion: public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("TransposeFusion");
    TransposeFusion(bool supports_immad = false);
};

class TransposeMatMulMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeMatMulMatcher");
    TransposeMatMulMatcher(bool supports_immad);
};

class TransposeMatMulTransposeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeMatMulTransposeMatcher");
    TransposeMatMulTransposeMatcher(bool supports_immad);
};

class TransposeSDPAMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeSDPAMatcher");
    TransposeSDPAMatcher();
};

}   // namespace ov::intel_gpu
