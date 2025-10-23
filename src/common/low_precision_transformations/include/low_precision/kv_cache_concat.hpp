// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "lpt_visibility.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::pass::low_precision {
/**
 * @brief Optimizes stateful KV cache by eliminating quantization-dequantization pairs on ReadValue-Assign paths.
 * 
 * This transformation identifies KV concatenation patterns where both cache and new KV data
 * undergo identical quantization (downconvert) followed by dequantization (upconvert). It optimizes
 * the pattern by:
 * 1. Storing the cache variable in low precision fp8 format
 * 2. Removing downconvert subgraphs from the cache branch
 * 3. Connecting the Assign operation directly to the low-precision concat output (so upconvert subgraph is also removed)
 */
class LP_TRANSFORMATIONS_API KVCacheConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KVCacheConcat");
    KVCacheConcat(const std::shared_ptr<ov::Model>& model);
};

} // namespace ov::pass::low_precision
