// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface InsertMoveBroadcast
 * @brief Inserts explicit MoveBroadcast instruction if broadcasting by most varying dimension is needed.
 * The pass is used to convert model to a canonical form for code generation
 * @ingroup snippets
 */
class InsertMoveBroadcast: public ov::pass::MatcherPass {
public:
    InsertMoveBroadcast();

    static Output<ov::Node> BroadcastNodeLastDim(const ov::Output<ov::Node>& value,
                                                     const ov::PartialShape& target_shape,
                                                     const ov::PartialShape& normalized_shape);
};

} // namespace pass
} // namespace snippets
} // namespace ov
