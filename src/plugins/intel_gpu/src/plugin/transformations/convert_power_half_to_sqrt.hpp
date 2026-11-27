// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

// Rewrites elementwise v1::Power(x, 0.5) (with an optional standard floating-point Convert on the constant)
// to v0::Sqrt(x) when output shape and element type are preserved, enabling downstream RMSFusion, L2NormToRMS,
// and Convert-wrapped MVNFusion patterns that match v0::Sqrt.
class ConvertPowerHalfToSqrt : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertPowerHalfToSqrt");
    ConvertPowerHalfToSqrt();
};

}  // namespace ov::intel_gpu
