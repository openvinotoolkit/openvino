// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::pass {

/**
 * @interface BroadcastToMoveBroadcast
 * @brief Inserts explicit MoveBroadcast instruction if broadcasting by most varying dimension is needed instead of
 * Broadcast. Otherwise the pass removes Broadcast operation.
 * @ingroup snippets
 */
class SNIPPETS_API BroadcastToMoveBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::BroadcastToMoveBroadcast");
    BroadcastToMoveBroadcast();
};

}  // namespace ov::snippets::pass
