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
 * @interface TransposeDecomposition
 * @brief Decompose Transpose to Load + Store wrapped in several loops.
 * @ingroup snippets
 */
class TransposeDecomposition: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeDecomposition", "0");
    TransposeDecomposition();
    static const std::set<std::vector<int>> supported_cases;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
