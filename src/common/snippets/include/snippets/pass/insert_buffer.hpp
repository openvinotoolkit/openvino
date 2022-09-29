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
 * @interface InsertBuffer
 * @brief The pass inserts Buffers on Inputs and Outputs of special operations [Softmax, Transpose] is it's needed
 * @ingroup snippets
 */
class InsertBuffer: public ngraph::pass::MatcherPass {
public:
    InsertBuffer(const int32_t allocation_rank = -1);
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
