// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class BroadcastReshapeMatmulFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("BroadcastReshapeMatmulFusion", "0");
    BroadcastReshapeMatmulFusion();
};

}   // namespace intel_gpu
}   // namespace ov
