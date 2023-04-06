// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface TransposeDecomposition
 * @brief Decompose Transpose to Load + Store wrapped in several loops.
 * @ingroup snippets
 */
class TransposeDecomposition: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeDecomposition", "0");
    TransposeDecomposition();
    static const std::set<std::vector<int>> supported_cases;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
