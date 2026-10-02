// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

// Prepares GEMM3_SWIGLU MOECompressed zero-points for the GPU kernels:
// - u3 zp is converted to u8, as ConvertFullyConnectedToFullyConnectedCompressed does for FC: the per-group
//   scale/zp reorders in prepare_quantization and the OneDNN grouped matmul zp buffers are byte-addressed.
// - a single-element u8/i8 zp shared by all experts is broadcast to the scale shape at compile time, since the
//   kernels and OneDNN grouped matmul (zp mask 7/5 only) take per-group zp.
class ConvertMOE3GemmZpToU8 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertMOE3GemmZpToU8");
    ConvertMOE3GemmZpToU8();
};

}  // namespace ov::intel_gpu
