// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class TransposeFusion: public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("TransposeFusion", "0", ov::pass::GraphRewrite);
    TransposeFusion(bool supports_immad = false);
};

class TransposeMatMulMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeMatMulMatcher", "0", ov::pass::MatcherPass);
    TransposeMatMulMatcher(bool supports_immad);
};

class TransposeMatMulTransposeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeMatMulTransposeMatcher", "0", ov::pass::MatcherPass);
    TransposeMatMulTransposeMatcher(bool supports_immad);
};

class TransposeSDPAMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeSDPAMatcher", "0", ov::pass::MatcherPass);
    TransposeSDPAMatcher();
};

}   // namespace intel_gpu
}   // namespace ov
