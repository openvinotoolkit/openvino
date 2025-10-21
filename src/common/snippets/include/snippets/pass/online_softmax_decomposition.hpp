// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface OnlineSoftmaxDecomposition
 * @brief Decomposes OnlineSoftmax to a range of low-level operations.
 * @ingroup snippets
 * scheme:
 * Decompose OnlineSoftmax to below subgraph:
 *  -----------Input
 *  |            |
 *  |        ReduceMax
 *  |            |
 *  |    OnlineSoftmaxUpdateMax
 *  |            |         |
 *  ----------Subtract     |
 *               |         |
 *  ------------Exp       Exp
 *  |            |         |
 *  |        ReduceSum     |
 *  |            |         |
 *  |    OnlineSoftmaxUpdateSum
 *  |            |\      |
 *  |            | \     |
 *  |            |  \    |
 *  |            |   \   |
 *  |          Power Divide
 *  |            |      |
 *  ----------Multiply  Result1
 *               |
 *            Result0
 */
class OnlineSoftmaxDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::OnlineSoftmaxDecomposition");
    OnlineSoftmaxDecomposition();
};

}  // namespace ov::snippets::pass
