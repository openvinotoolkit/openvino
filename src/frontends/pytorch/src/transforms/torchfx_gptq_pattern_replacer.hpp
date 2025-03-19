// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

// This transformation replaces the GPTQ pattern with a Constant node
class GPTQDecompressionReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::GPTQDecompressionReplacer");
    GPTQDecompressionReplacer();
};

// This transformation replaces TorchFX based GPTQ Multiplication pattern
// into a simpler pattern which could be detected by device plugins for
// additional optimizations
class GPTQMultPatternReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::GPTQMultPatternReplacer");
    GPTQMultPatternReplacer();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
