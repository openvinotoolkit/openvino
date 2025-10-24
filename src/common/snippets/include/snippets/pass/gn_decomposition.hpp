// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::pass {

/**
 * @interface GNDecomposition
 * @brief Decomposes GroupNormalization to a range of low-level operations
 * @ingroup snippets
 */
class SNIPPETS_API GNDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::GNDecomposition");
    GNDecomposition();
};

}  // namespace ov::snippets::pass
