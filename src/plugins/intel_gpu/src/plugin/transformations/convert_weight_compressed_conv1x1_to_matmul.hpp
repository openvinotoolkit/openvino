// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class ConvertWeightCompressedConv1x1ToMatmul : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertWeightCompressedConv1x1ToMatmul");
    ConvertWeightCompressedConv1x1ToMatmul(bool supports_immad = false);
};

class ConvertWeightCompressedConv1x1ToMatmulMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertWeightCompressedConv1x1ToMatmulMatcher");
    ConvertWeightCompressedConv1x1ToMatmulMatcher(bool supports_immad);
};

}  // namespace ov::intel_gpu
