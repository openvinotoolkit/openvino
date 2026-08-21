// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Optimization pass for the prefill model: instead of running the full last
// transformer layer for all N query tokens and slicing the output after the
// LM head (SliceOutEmbeds), slice Q to the last K positions *before* the
// last layer's ScaledDotProductAttention (K = num_last_tokens, typically 1).
//
// This saves attention, o_proj, and FFN computation for N-K positions in the
// last layer.  K and V are not sliced — full N-token KV is needed for causal
// attention.
//
// Must be called BEFORE DecomposeGQA (which replaces the native SDPA op).
// Works on the static-shape prefill model after ReshapeToStatic.
class SliceLastTokenPrefill : public ov::pass::ModelPass {
    uint32_t m_batch_dim;
    uint32_t m_num_last_tokens;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::SliceLastTokenPrefill");
    explicit SliceLastTokenPrefill(uint32_t batch_dim, uint32_t num_last_tokens = 1);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
