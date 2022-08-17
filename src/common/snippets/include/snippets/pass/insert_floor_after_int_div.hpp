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
 * @interface InsertFloorAfterIntDiv
 * @brief Inserts explicit floor instruction after each div with integer inputs
 *        to align with python_div behavior
 * @ingroup snippets
 */
class InsertFloorAfterIntDiv: public ngraph::pass::MatcherPass {
public:
    InsertFloorAfterIntDiv();
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
