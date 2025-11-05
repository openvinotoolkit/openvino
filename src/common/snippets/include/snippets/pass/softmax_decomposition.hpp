// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::pass {

/**
 * @interface SoftmaxDecomposition
 * @brief Decomposes Softmax to a range of low-level operations
 * @ingroup snippets
 */
class SNIPPETS_API SoftmaxDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::SoftmaxDecomposition");
    SoftmaxDecomposition();
};

}  // namespace ov::snippets::pass
