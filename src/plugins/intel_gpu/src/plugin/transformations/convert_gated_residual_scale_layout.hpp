// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// @brief Converts the Constant scales / zero points of a compressed GatedResidual READ from the operation layout [N, G]
/// to the [G, N] layout the GPU kernels read (N contiguous: one sub-group block read gives every lane the scale of its own
/// output channel). The transposed constants are folded at compile time and the op is re-created with `scale_gn` set.
class ConvertGatedResidualScaleLayout : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGatedResidualScaleLayout");
    ConvertGatedResidualScaleLayout();
};

}  // namespace ov::intel_gpu
