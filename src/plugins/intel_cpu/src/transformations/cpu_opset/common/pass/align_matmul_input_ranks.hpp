// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

/*
 * Description:
 *     AlignMatMulInputRanks transformation detects MatMul operations
 *     and unsqueezes one input to another to align the ranks of the inputs.
 *     The transformation is required because oneDNN library
 *     requires inputs to have equal ranks
 */

namespace ov::intel_cpu {

class AlignMatMulInputRanks : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("AlignMatMulInputRanks");
    AlignMatMulInputRanks();
};

}  // namespace ov::intel_cpu
