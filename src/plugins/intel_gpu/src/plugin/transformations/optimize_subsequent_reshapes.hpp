// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

/**
 * @brief This pass looks for `Reshape [ dynamic dim, n static dims] -> Reshape [dynamic dim, n static dims]` patterns
 *        and replaces them with a single `Reshape [dynamic dim, n static dims]` operation.
 */
class OptimizeSubsequentReshapes : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("OptimizeSubsequentReshapes");
    OptimizeSubsequentReshapes();
};

}   // namespace intel_gpu
}   // namespace ov
