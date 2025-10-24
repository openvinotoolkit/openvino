// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

// Fuse subgraph between router and MOECompressed into MOECompressed operation, get a MOEFusedCompressed operation.
class FuseMOECompressed: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMOECompressed");
    FuseMOECompressed();
};

}   // namespace ov::intel_gpu
