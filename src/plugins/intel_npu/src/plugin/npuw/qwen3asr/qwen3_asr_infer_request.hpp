// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../llm_infer_request.hpp"

namespace ov {
namespace npuw {

// Dedicated infer request for Qwen3-ASR encoder-decoder LLM models.
// Extends LLMInferRequest with:
//   - Qwen3-ASR-specific infer() dispatch (seq_len > 1 → prefill, seq_len == 1 → generate)
//   - infer_prefill(): right-aligned token copy + attention_mask/position_ids injection + encoder_hidden_states
//   - infer_generate(): O(1) per-step decoding with static KV cache, attention_mask, position_ids
class Qwen3ASRInferRequest final : public LLMInferRequest {
public:
    explicit Qwen3ASRInferRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model);

    void infer() override;

protected:
    void prepare_for_new_conversation() override;

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                       ov::SoPtr<ov::ITensor> enc_hidden_states);

    void infer_generate(ov::SoPtr<ov::ITensor> input_ids);
};

}  // namespace npuw
}  // namespace ov
