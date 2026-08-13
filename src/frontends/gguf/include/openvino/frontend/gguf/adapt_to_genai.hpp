// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace pass {

/// \brief Rewrite a GGUF-frontend model's llama.cpp-style IO into the OpenVINO GenAI
///        LLMPipeline IO contract, so the model can be driven by genai's stateful pipeline.
///
/// The GGUF frontend emits a stateful decoder with the gguf IO contract:
///   inputs : inp_tokens [1,1,1,D] i32, inp_pos [1,1,1,D] i32, inp_out_ids [1,1,1,D] i32,
///            self_kq_mask [1,1,D,D] f32 (+ self_kq_mask_swa for gpt-oss SWA),
///            token_len_per_seq [1] i64, plus beam_idx [D] i32 from the make-stateful pass
///   output : logits [1,1,seq,vocab]
///
/// genai's StatefulLLMPipeline instead feeds:
///   inputs : input_ids [b,seq] i64, attention_mask [b,kv_len] i64,
///            position_ids [b,seq] i64, beam_idx [b] i32
///   output : logits [b,seq,vocab]
///
/// This pass prepends a subgraph deriving the gguf inputs from the genai ones, rewires the gguf
/// Parameters to it, and reshapes the logits to [b,seq,vocab]. The model must already be stateful:
/// the KV-cache sinks are preserved, and beam_idx (created by the make-stateful pass) passes
/// through unchanged since genai sets that tensor itself.
///
/// If the required gguf inputs are absent (e.g. the model is already in genai form), the
/// pass is a no-op and returns false.
///
/// LAYOUT POLYMORPHISM (why the leading dims are derived, never pinned).
/// The result must be valid under both attention backends, which disagree about where the token
/// count lives: plain SDPA feeds input_ids as [1, tokens], while ov::pass::SDPAToPagedAttention
/// rewrites it to rank-1 [tokens] and unsqueezes, so the body sees [tokens, 1]. Both are the same
/// buffer -- ggml's layout is [batch, tokens, heads, head_size] with batch == 1 -- so one graph
/// serves both PROVIDED no node pins the leading two dims. Hence deriving them from the live
/// input_ids here, and reshaping with special_zero in the translators. No backend flag needed.
class GGUF_FRONTEND_API AdaptToGenAI : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::AdaptToGenAI");

    /// \brief Which genai input contract to expose.
    /// IdsToLogits  : input_ids -> logits (text LLMPipeline). The only mode implemented today.
    /// EmbedsToLogits: inputs_embeds -> logits (reserved for the VLM language model, where
    ///                 image+text embeddings are merged outside the graph). Not yet implemented.
    enum class InputMode { IdsToLogits, EmbedsToLogits };

    explicit AdaptToGenAI(InputMode mode = InputMode::IdsToLogits) : m_mode(mode) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    InputMode m_mode;
};

}  // namespace pass
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
