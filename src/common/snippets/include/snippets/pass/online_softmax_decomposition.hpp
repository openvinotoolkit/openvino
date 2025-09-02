// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface OnlineSoftmaxDecomposition
 * @brief Decomposes OnlineSoftmaxDecomposition to a range of low-level operations.
 * @ingroup snippets
 */
class OnlineSoftmaxDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::OnlineSoftmaxDecomposition");
    OnlineSoftmaxDecomposition();
};

}  // namespace ov::snippets::pass
