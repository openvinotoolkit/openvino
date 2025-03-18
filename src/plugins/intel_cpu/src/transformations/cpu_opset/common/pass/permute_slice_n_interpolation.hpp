// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

// SS(NHWC)->Transpose(fake)->Interpolate(NHWC as NCHW)
// NHWC->Interpolate(NHWC as NCHW)-NCHW->Transpose(fake)->SS

namespace ov::intel_cpu {

class PermuteSliceAndInterpolation : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PermuteSliceAndInterpolation");
    PermuteSliceAndInterpolation();
};

}  // namespace ov::intel_cpu
