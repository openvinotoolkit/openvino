// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// @brief Compile-time "weight shuffle" repack of Q4_K / Q5_K / Q6_K and small-block Q4_0 / Q4_1 /
/// Q8_0 GGUF FullyConnected weights into the SG-transposed layout consumed by the sub-group-block-read
/// GEMV kernels (fc_gguf_gemv_opt.cl) and the shuffle-aware prefill transcode (fc_gguf_transcode.cl).
/// Small-block formats (native 32-elem block) are grouped 8-to-a 256-elem super-block first.
///
/// The repack is a two-step, value-preserving reorder of the weight Constant (no dequant):
///   1) shuffle: re-encode each native GGUF block into the plane-separated row-major layout
///        Q4_K -> pqs[128] (nibble-repacked quants) + psl[16] (sl/ml/sh/mh + d + dmin)
///        Q6_K -> pql[128] + pqh[64] + ps[16] + pd[2]
///   2) SG pack: group N-rows by SG=16 and interleave the 16 blocks of each K-block in 4-byte chunks
///        so a single intel_sub_group_block_read delivers coalesced weight bytes to the 16 lanes.
///
/// The shuffled representation keeps the SAME element type (gguf_q4_k / gguf_q6_k), the SAME logical
/// Shape{N, K}, and the SAME byte size (Q4_K 144 B/block, Q6_K 210 B/block are preserved). The GEMV
/// and transcode kernels decode it bit-exactly.
///
/// With defer_to_gpu=true, the replacement Constant shares the immutable raw storage and carries
/// a pending-shuffle marker. Constant upload performs the GPU reorder and releases its temporary
/// raw buffer after completion. The default CPU mode remains available as a byte-exact oracle.
/// Prefill and decode share the uploaded shuffled weight without a first-token repack.
///
/// Gated on: weight is a shuffle-eligible GGUF format (Q4_K/Q5_K/Q6_K or Q4_0/Q4_1/Q8_0) AND
/// N % 16 == 0 (SG grouping) AND K % 256 == 0. The same formula on the same (dtype, N, K) keeps the
/// transform and FCGGUFOptImpl in lockstep.
class RepackGGUFWeightsShuffle : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RepackGGUFWeightsShuffle");
    explicit RepackGGUFWeightsShuffle(bool defer_to_gpu = false);
};

}  // namespace ov::intel_gpu
