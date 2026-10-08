// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class ConvertFullyConnectedToFullyConnectedCompressed: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertFullyConnectedToFullyConnectedCompressed");
    // keep_u2_group_major: when the device can run the CM u2 group-major kernel, leave a u2 FC whose
    // weights are a 3-D group-major [KG, N, GS] constant un-transposed (no weight transpose folded in),
    // so the FC is fed the native group-major layout the kernel reads directly (WLAYOUT 0).
    explicit ConvertFullyConnectedToFullyConnectedCompressed(bool keep_u2_group_major = false);
};

}   // namespace ov::intel_gpu
