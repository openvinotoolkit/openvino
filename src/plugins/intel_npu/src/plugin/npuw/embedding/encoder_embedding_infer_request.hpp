// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../base_sync_infer_request.hpp"
#include "../llm_compiled_model.hpp"
#include "../llm_infer_base_request.hpp"
#include "openvino/core/descriptor/output.hpp"

namespace ov {
namespace npuw {

// Infer request for non-autoregressive bidirectional encoder (e.g. BERT) text-embedding models.
// A single static forward over the whole (right-padded) sequence, with no KV cache and no
// chunking. No position_ids are injected: the encoder works out its own positions, so this path
// does not care whether they are the learned absolute ones that BERT, RoBERTa and XLM-R use or
// the RoPE and ALiBi schemes some newer encoders have moved to. Compare EmbeddingInferRequest,
// which targets the autoregressive (Qwen3-Embedding-style) reconstructed prefill model.
class EncoderEmbeddingInferRequest : public ov::npuw::LLMInferBaseRequest {
public:
    explicit EncoderEmbeddingInferRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model);

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

private:
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_in_ports;

    std::shared_ptr<ov::IAsyncInferRequest> m_prefill_request;

    // The compiled encoder's own output tensor, returned to the caller unchanged: it is already
    // sized [1, max_prompt_size, hidden], so there is nothing to copy it into.
    ov::SoPtr<ov::ITensor> m_prefill_out_tensor;
};

}  // namespace npuw
}  // namespace ov
