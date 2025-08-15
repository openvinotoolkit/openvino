// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface FADecomposition
 * @brief Decomposes attention pattern(MM + softmax + MM) to a range of low-level operations with falsh attention manner
 * @ingroup snippets
 */
class FADecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::FADecomposition");
    FADecomposition();
};

}  // namespace ov::snippets::pass
