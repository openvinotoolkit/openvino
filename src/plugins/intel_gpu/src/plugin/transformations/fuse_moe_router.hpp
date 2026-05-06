// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// Fuse routing subgraph (Softmax/Sigmoid+bias → TopK → Normalize → Transpose → Unsqueeze)
/// into MoERouterFused, connecting its outputs directly to MOECompressed inputs.
class FuseMoERouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoERouter");
    FuseMoERouter();
};

}  // namespace ov::intel_gpu
