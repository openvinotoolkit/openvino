// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "lpt_visibility.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::pass::low_precision {
/**
 * @brief Moves FakeConvert operations from after concat to before concat in KV cache patterns.
 * 
 * This transformation is a prerequisite for KV cache quantization optimization. It identifies
 * patterns where FakeConvert operations are applied to the output of KV cache concatenation
 * and moves them to the individual concat inputs instead.
 */
class LP_TRANSFORMATIONS_API MoveFakeConvertUpThroughKVCacheConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveFakeConvertUpThroughKVCacheConcat");
    MoveFakeConvertUpThroughKVCacheConcat();
};

}  // namespace ov::pass::low_precision
