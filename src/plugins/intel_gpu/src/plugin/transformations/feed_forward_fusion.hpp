// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class FeedForwardFusion: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FeedForwardFusion");
    FeedForwardFusion();
};

}   // namespace ov::intel_gpu
