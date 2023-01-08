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
 * @interface InsertMoveBroadcast
 * @brief Inserts explicit MoveBroadcast instruction if broadcasting by most varying dimension is needed.
 * The pass is used to convert model to a canonical form for code generation
 * @ingroup snippets
 */
class InsertMoveBroadcast: public ngraph::pass::MatcherPass {
public:
    InsertMoveBroadcast();
};

} // namespace pass
} // namespace snippets
} // namespace ngraph