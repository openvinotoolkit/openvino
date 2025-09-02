// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class TransposeFusion : public ov::pass::GraphRewrite {
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

class TransposeVLSDPAMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeVLSDPAMatcher");
    TransposeVLSDPAMatcher();
};
class TransposeConv1x1TransposeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeConv1x1TransposeMatcher");
    TransposeConv1x1TransposeMatcher(bool supports_immad);
};
class TransposeConv1x1ConvertTransposeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TransposeConv1x1ConvertTransposeMatcher");
    TransposeConv1x1ConvertTransposeMatcher(bool supports_immad);
};
class ReshapeConv1x1ReshapeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReshapeConv1x1ReshapeMatcher");
    ReshapeConv1x1ReshapeMatcher(bool supports_immad);
};
class ReshapeConv1x1ConvertReshapeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ReshapeConv1x1ConvertReshapeMatcher");
    ReshapeConv1x1ConvertReshapeMatcher(bool supports_immad);
};

}  // namespace ov::intel_gpu