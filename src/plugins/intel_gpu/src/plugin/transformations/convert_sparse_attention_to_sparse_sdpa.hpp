// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Lowers ov::op::v17::SparseAttention to the GPU SparseSDPA op.
///
/// Both read the dense key / value history with left padding (logical key j of row b at index S - key_length[b] + j)
/// and attend causally to the selected complete blocks plus the incomplete block of each query. SparseSDPA needs:
///  - static query / key-value / selection head counts and head sizes, which become its Config;
///  - query_block_size == 1;
///  - i32 block_indices / block_count, converted when they are i64;
///  - an i32 [B] key_length: an i64 one is converted, a [1] one broadcast over the batch, and an absent one is the
///    key sequence length of every row.
/// The optional scale and sinks inputs are passed through; the GPU lowering of SparseSDPA reports the forms its kernels
/// do not support (a non-constant scale, sinks). SparseAttention that cannot be expressed as SparseSDPA is left as is.
class ConvertSparseAttentionToSparseSDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertSparseAttentionToSparseSDPA");
    ConvertSparseAttentionToSparseSDPA();
};

}  // namespace ov::intel_gpu
