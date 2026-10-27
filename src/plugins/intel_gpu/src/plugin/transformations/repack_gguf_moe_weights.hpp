// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// @brief Compile-time repack of native GGUF MoE expert weights (Q4_K / Q5_K / Q6_K only) into the
/// transposed "SG" (sub-group block-read) layout consumed by the high-performance MoE decode GEMV
/// kernels in moe_gguf_sg_gemv.cl (see that file's header comment for the full byte layout and
/// q4k_moe_gemv/test_moe_gemv_sg_kernels.py for the reference implementation this mirrors).
///
/// This is a PURE BYTE PERMUTATION — same element type, same logical shape, same byte size (0%
/// expansion), decoded bit-exactly. Every other GGUF weight compression type (Q4_0, Q8_0) is left
/// untouched in its raw per-row GGUF-block layout: Q8_0 is decoded directly from that layout by
/// moe_gguf_sg_gemv.cl's dedicated shared-expert kernels, and the raw-GGUF-block batched-GEMV
/// kernels (moe_3gemm_swiglu_mlp.cl) handle any remaining case.
///
/// Done here (replacing the weight Constant at compile_model) so every decode call shares one
/// already-packed weight — single resident copy, no first-decode repack stall. Gated on the same
/// env var the impl uses to select the SG kernels (OV_GPU_GGUF_MOE_SG, default on) plus N % 16 == 0
/// (Q4_K/Q5_K/Q6_K packing group size) and K being a whole number of 256-element super-blocks.
class RepackGGUFMoEWeights : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RepackGGUFMoEWeights");
    RepackGGUFMoEWeights();
};

}  // namespace ov::intel_gpu
