// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface InsertRoundingAfterIntDiv
 * @brief Inserts explicit round instruction after each div with integer inputs
 *        to align with the corresponding behavior:
 *        - floor : to -INF aligned by python div
 *        - trunc : to zero aligned by c++ div
 * @ingroup snippets
 */
class InsertRoundingAfterIntDiv: public ngraph::pass::MatcherPass {
public:
    InsertRoundingAfterIntDiv();
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
