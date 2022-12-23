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
 * @interface BroadcastToMoveBroadcast
 * @brief Inserts explicit MoveBroadcast instruction if broadcasting by most varying dimension is needed instead of Broadcast.
 *        Otherwise the pass removes Broadcast operation.
 * @ingroup snippets
 */
class BroadcastToMoveBroadcast: public ngraph::pass::MatcherPass {
public:
    BroadcastToMoveBroadcast();
};


} // namespace pass
} // namespace snippets
} // namespace ngraph