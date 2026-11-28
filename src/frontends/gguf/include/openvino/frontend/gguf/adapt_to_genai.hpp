// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/model_adapter.hpp"
#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::gguf::pass {

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
/// Gated-DeltaNet recurrent states get a dynamic batch dimension and follow beam_idx; models with
/// other recurrent states (Mamba 2) keep input_ids at one sequence.
///
/// token_len_per_seq is optional when the source graph does not consume it.
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
    /// IDS_TO_LOGITS  : input_ids -> logits (text LLMPipeline).
    /// EMBEDS_TO_LOGITS: raw inputs_embeds -> logits, with scaling retained in the decoder.
    /// Token-dependent auxiliary branches retain input_ids when required. Per-layer token
    /// embeddings (Gemma4 E2B/E4B) move to a separate lookup model and become the
    /// per_layer_inputs [B,T,layers,width] input; Gemma3 and other Gemma4 models take
    /// token_type_ids [B,T] for bidirectional attention within each image. M-RoPE models
    /// receive position_ids [3,B,T] or [4,B,T], depending on the model's layout.
    /// EMBEDS_TO_LOGITS_TEXT uses scalar text position_ids [B,T], including for M-RoPE models.
    enum class InputMode { IDS_TO_LOGITS, EMBEDS_TO_LOGITS, EMBEDS_TO_LOGITS_TEXT };

    explicit AdaptToGenAI(InputMode mode = InputMode::IDS_TO_LOGITS) : m_mode(mode) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    /// Raw token lookup extracted by either embedding mode, sharing the original weight buffers.
    /// Available after a successful run; inputs input_ids [B,T], output inputs_embeds [B,T,D].
    const std::shared_ptr<ov::Model>& get_embedding_model() const {
        return m_embedding_model;
    }

    /// Per-layer token lookup extracted by EMBEDS_TO_LOGITS (Gemma4 E2B/E4B), or null.
    /// Inputs input_ids [B,T], output per_layer_inputs [B,T,layers,width].
    const std::shared_ptr<ov::Model>& get_per_layer_embedding_model() const {
        return m_per_layer_embedding_model;
    }

    /// Hashed n-gram lookup extracted by EMBEDS_TO_LOGITS (Qwen3.8-Flash-Next), or null.
    /// Inputs input_ids [B,T] and ngram_history [B,n-1] (absolute token ids, EOS for a new
    /// sequence); outputs ngram_embeddings [B,T,1,width] and new_ngram_history [B,n-1].
    const std::shared_ptr<ov::Model>& get_ngram_embedding_model() const {
        return m_ngram_embedding_model;
    }

private:
    InputMode m_mode;
    std::shared_ptr<ov::Model> m_embedding_model;
    std::shared_ptr<ov::Model> m_per_layer_embedding_model;
    std::shared_ptr<ov::Model> m_ngram_embedding_model;
};

}  // namespace ov::frontend::gguf::pass

namespace ov::frontend::gguf {

struct GenAIAdaptOptions {
    /// Input contract of a decoder's language model; ignored for an mmproj.
    pass::AdaptToGenAI::InputMode input = pass::AdaptToGenAI::InputMode::IDS_TO_LOGITS;
};

/// \brief Adapt a model returned by FrontEnd::convert to the models used by OpenVINO GenAI.
///
/// Convert without GenAI-specific extensions, then call this once per converted file:
///
///     ov::frontend::gguf::FrontEnd fe;
///     auto models = ov::frontend::gguf::adapt_to_genai(fe.convert(fe.load(gguf_path)));
///     auto language = models.at("language");
///
/// A decoder is made stateful (GGUFMakeStateful) and adapted (AdaptToGenAI). It returns
/// "language" and, for either embedding mode, "text_embeddings" plus "text_embeddings_per_layer"
/// when the architecture has per-layer token embeddings.
/// A decoder loaded with LoadOptions::include_mtp_draft also returns "mtp_draft" when
/// the file contains a supported NextN block. A file that holds only an MTP head returns
/// "mtp_draft" alone. The returned models share weight buffers.
///
/// An mmproj returns one entry per encoder:
///  - "vision_embeddings" with the GenAI vision encoder inputs, plus "vision_embeddings_pos" and
///    "vision_embeddings_merger" for split-merger projectors; projector callbacks name their own
///    models. A vision projector without a GenAI layout keeps the native encoder inputs.
///  - "audio_embeddings"; Gemma4 encoders take GenAI input_features, other encoders keep their
///    native inputs.
///  - "audio_generation_<component>" for each declared audio-generation component.
///
/// The supplied model is consumed: the returned models may reuse or modify it, and share its
/// weights. Throws if the model is neither a converted GGUF decoder nor an mmproj.
GGUF_FRONTEND_API ModelMap adapt_to_genai(const std::shared_ptr<ov::Model>& model,
                                          const GenAIAdaptOptions& options = {});

}  // namespace ov::frontend::gguf
