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
 * @interface SoftmaxReshapeElimination
 * @brief The pass removes Reshape operations around Softmax if possible
 * @ingroup snippets
 */
class SoftmaxReshapeElimination: public ngraph::pass::MatcherPass {
public:
    SoftmaxReshapeElimination();
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
