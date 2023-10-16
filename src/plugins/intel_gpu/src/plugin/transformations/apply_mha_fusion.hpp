// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/core/visibility.hpp"

namespace ov {
namespace intel_gpu {

/**
 * @brief Apply Multi-Head Attention fusion to improve performance.
 * Matching pattern : (Matrices Q,K,V)
 *                       Q   K
 *                       |  /
 *                     MatMul
 *                       |
 *                    Softmax  V
 *                       |    /
 *                       |   /
 *                     Matmul
 */
class ApplyMHAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ApplyMHAFusion", "0");
    ApplyMHAFusion();
};

}  // namespace intel_gpu
}  // namespace ov
