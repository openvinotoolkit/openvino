// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

/*
 * Description:
 *     AlignMatMulInputRanks transformation detects MatMul operations
 *     and unsqueezes one input to another to align the ranks of the inputs.
 *     The transformation is required because oneDNN library
 *     requires inputs to have equal ranks
 */

namespace ov {
namespace intel_cpu {

class AlignMatMulInputRanks: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("AlignMatMulInputRanks", "0");
    AlignMatMulInputRanks();
};

}   // namespace intel_cpu
}   // namespace ov
