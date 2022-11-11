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
 * @interface MatMulTranspose
 * @brief At the moment Snippets supports Transpose only with order {0, 2, 3, 1},
 *        so if there is pattern in graph:
 *         in0     Transpose{0, 2, 1, 3}
 *           \    /
 *           MatMul[false, true]
 *        We can set false in MatMul parameter `transposed_b` and
 *        change Transpose order to {0, 2, 3, 1} which is supported by Snippets
 * @ingroup snippets
 */
class MatMulTranspose: public ngraph::pass::MatcherPass {
public:
    MatMulTranspose();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
