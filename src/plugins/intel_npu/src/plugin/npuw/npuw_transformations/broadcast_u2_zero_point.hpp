// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::npuw {

// Broadcasts a scalar zero point in the dequantization chain of u2 weights to a per-group u2 Constant:
//   Convert(Constant u2 [..., groups, group_size]) -> Subtract <- Constant (floating-point scalar)
// becomes
//   Convert(Constant u2 [..., groups, group_size]) -> Subtract <- Convert(Constant u2 [..., groups, 1])
// The rewrite is applied only when the zero point is an integer in the range of u2, so the numerical result is
// unchanged, and the zero point gets the same element type and the group shape of the weights scales.
class BroadcastU2ZeroPoint : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ov::npuw::BroadcastU2ZeroPoint");
    BroadcastU2ZeroPoint();
};

}  // namespace ov::npuw
