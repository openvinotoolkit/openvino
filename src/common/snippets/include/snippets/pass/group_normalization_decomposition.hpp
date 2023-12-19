// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface GroupNormalizationDecomposition
 * @brief Decomposes GroupNormalization to a range of low-level operations
 * @ingroup snippets
 */
class GroupNormalizationDecomposition: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GroupNormalizationDecomposition", "0");
    GroupNormalizationDecomposition();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
