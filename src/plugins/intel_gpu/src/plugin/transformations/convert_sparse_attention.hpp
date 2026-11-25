// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string>

#include "intel_gpu/op/sparse_sdpa.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// Converts ov::op::v17::SparseAttention (dense K / V history, no paging) to the GPU SparseSDPA op, the only form lowered
/// to cldnn::sparse_sdpa: IndirectKVCache and KVCacheCompression then only handle SparseSDPA.
///
/// SparseSDPA reads i32 selections and one i32 key_length per batch row, so the selections are converted and
/// key_length is normalized here: an absent key_length is the key sequence length of every row, a scalar or [1] one is
/// broadcast over the batch. The head geometry comes from the input shapes; the scale and the sinks are passed on and
/// checked by the lowering (CreateSparseSDPAOp).
///
/// An op whose head geometry is not static, or whose configuration SparseSDPA does not support, is left as is so that a
/// model query reports it; its lowering (CreateSparseAttentionOp) then fails with the reason.
class ConvertSparseAttentionToInternal : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertSparseAttentionToInternal");
    ConvertSparseAttentionToInternal();

    /// The SparseSDPA configuration of `op`, or nullopt with the `reason` it cannot be converted.
    static std::optional<op::SparseSDPA::Config> make_config(const ov::op::v17::SparseAttention& op, std::string& reason);
};

}  // namespace ov::intel_gpu
