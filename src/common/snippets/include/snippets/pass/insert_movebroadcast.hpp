// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface InsertMoveBroadcast
 * @brief Inserts explicit MoveBroadcast instruction if broadcasting by most varying dimension is needed.
 * The pass is used to convert model to a canonical form for code generation
 * @ingroup snippets
 */
class InsertMoveBroadcast : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::InsertMoveBroadcast");
    InsertMoveBroadcast();

    static Output<ov::Node> BroadcastNodeLastDim(const ov::Output<ov::Node>& value,
                                                 const ov::PartialShape& target_shape,
                                                 const ov::PartialShape& normalized_shape);
};

}  // namespace ov::snippets::pass
