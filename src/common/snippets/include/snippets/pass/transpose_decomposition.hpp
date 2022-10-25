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
 * @interface InsertLoops
 * @brief Insert explicit Loop operations into the body to process multiple data entities during one kernel execution
 * @ingroup snippets
 */
class TransposeDecomposition: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeDecomposition", "0");
    TransposeDecomposition();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
