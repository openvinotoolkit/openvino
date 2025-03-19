// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface GNDecomposition
 * @brief Decomposes GroupNormalization to a range of low-level operations
 * @ingroup snippets
 */
class GNDecomposition: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::GNDecomposition");
    GNDecomposition();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
