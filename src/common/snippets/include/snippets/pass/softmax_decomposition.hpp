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
 *        Note:
 *            - At the moment Snippets supports Softmax only in MHA pattern where there are Buffer ops before and after Softmax.
 *              Also Snippets support Loops with Buffer ops on inputs and outputs if Buffer have the same buffer byte size
 *              because of work with ptr increment. So we have to set Tile rank as buffer allocation rank even if rank 1 is enough
 * @ingroup snippets
 */
class SoftmaxDecomposition: public ngraph::pass::MatcherPass {
public:
    SoftmaxDecomposition(const size_t vector_size, const int32_t buffer_allocation_rank = -1);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
