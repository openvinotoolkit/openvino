// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// @brief Lower a generic ov::op::internal::FullyConnectedCompressed whose weight is an opaque GGUF
/// block type into the GPU-internal ov::intel_gpu::op::FullyConnectedCompressed so the GPU plugin can
/// create a cldnn primitive for it.
///
/// The native GGUF FrontEnd emits ov::op::internal::FullyConnectedCompressed directly (weight is a
/// gguf_* Constant, scale/zero-point inputs empty), so it never goes through the
/// MatMul -> FullyConnected -> compress chain that ConvertFullyConnectedToFullyConnectedCompressed
/// rewrites. This pass performs the remaining 1:1 op-type lowering for the GGUF case only; all other
/// (non-GGUF) FullyConnectedCompressed nodes are produced by the GPU op directly and are left
/// untouched.
class ConvertGGUFFullyConnectedCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGGUFFullyConnectedCompressed");
    ConvertGGUFFullyConnectedCompressed();
};

}  // namespace ov::intel_gpu
