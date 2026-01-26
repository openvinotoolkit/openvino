// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

// Fuse subgraph between router and MOECompressed into MOE3GemmFusedCompressed operation.
class FuseMOE3GemmCompressed: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMOE3GemmCompressed");
    FuseMOE3GemmCompressed();
};

}   // namespace ov::intel_gpu
