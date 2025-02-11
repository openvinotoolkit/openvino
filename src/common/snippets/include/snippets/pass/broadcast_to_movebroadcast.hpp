// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface BroadcastToMoveBroadcast
 * @brief Inserts explicit MoveBroadcast instruction if broadcasting by most varying dimension is needed instead of Broadcast.
 *        Otherwise the pass removes Broadcast operation.
 * @ingroup snippets
 */
class BroadcastToMoveBroadcast: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::BroadcastToMoveBroadcast");
    BroadcastToMoveBroadcast();
};


} // namespace pass
} // namespace snippets
} // namespace ov
