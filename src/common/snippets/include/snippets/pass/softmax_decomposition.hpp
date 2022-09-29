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
 * @interface SoftmaxDecomposition
 * @brief The pass decomposise Softmax into explicit Snippets dialects
 * @ingroup snippets
 */
class SoftmaxDecomposition: public ngraph::pass::MatcherPass {
public:
    SoftmaxDecomposition(const size_t vector_size);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
