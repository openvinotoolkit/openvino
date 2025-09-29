// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/core/visibility.hpp"

namespace ov::intel_gpu {

/**
 * @brief Decomposes NormalizeL2 into subgraph
 */
class NormalizeL2Decomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NormalizeL2DecompositionGPU");
    NormalizeL2Decomposition(bool use_fp32_reducesum = false);
};

}  // namespace ov::intel_gpu
