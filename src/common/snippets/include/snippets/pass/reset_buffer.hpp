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
 * @interface ResetBufferState
 * @brief If there is Buffer between loops we should reset Buffer pointer after first loop execution (data storing) using finalization offsets
 *        to have correct buffer data pointer for data loading in the next loop where data was stored in previous loop
 * @ingroup snippets
 */
class ResetBufferState: public ngraph::pass::MatcherPass {
public:
    ResetBufferState();

    static int64_t calculate_required_finalization_offsets(const size_t inner_master_work_amount, const size_t inner_target_work_amount);
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
