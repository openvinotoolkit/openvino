// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

// SS(NHWC)->Transpose(fake)->Interpolate(NHWC as NCHW)
// NHWC->Interpolate(NHWC as NCHW)-NCHW->Transpose(fake)->SS

namespace ov {
namespace intel_cpu {

class PermuteSliceAndInterpolation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PermuteSliceAndInterpolation", "0");
    PermuteSliceAndInterpolation();
};

}   // namespace intel_cpu
}   // namespace ov
