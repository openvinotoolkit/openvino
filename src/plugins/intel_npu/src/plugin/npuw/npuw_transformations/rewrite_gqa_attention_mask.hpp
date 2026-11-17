// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Rewrites the softmax mask produced by a decomposed GroupQueryAttention op so that
// it honours the model's `attention_mask` input (needed for NPUW static, padded KV buffers).
//
// Matched pattern (the Select's friendly name must contain "GroupQueryAttention"):
//
//   CumSum(K) ----------------------------------------------+
//                                                           v
//   CumSum(Q) -> Add -> Broadcast -> Transpose ---------> Less -> Select -> Add -> [Maximum] -> Softmax
//
// Rewritten to:
//
//   CumSum(K) ------------------------------------+
//                                                 v
//   CumSum(Q) -> Broadcast -> Transpose -----> LessEqual -> LogicalAnd -> Select -> ...
//                                                              ^
//   attention_mask -> Convert(boolean) -> [Unsqueeze] ---------+
//
// Run after static reshaping. If the mask's penultimate dimension is greater than 1,
// remove the original Add and use LessEqual AND attention_mask (prefill).
// If it equals 1, connect the boolean attention_mask directly to Select (KVCache).
class RewriteGQAAttentionMask : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::RewriteGQAAttentionMask");
    RewriteGQAAttentionMask() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
