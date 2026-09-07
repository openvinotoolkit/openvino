// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <string>

#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::gguf::pass {

/// \brief Turn the frontend's stateless GGUF model into an OpenVINO stateful one.
///
/// The frontend always converts to a STATELESS graph -- every KV cache is a Parameter written by a
/// SetRows placeholder and read back as a Result -- mirroring how optimum-intel exports. Being
/// stateful is a caller concern, so register this as a DecoderTransformationExtension:
///
///     ov::frontend::gguf::FrontEnd fe;
///     fe.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
///         ov::frontend::gguf::pass::GGUFMakeStateful()));
///     auto model = fe.convert(fe.load(decoder_or_gguf_path));
///
/// Extensions run ahead of the built-in LowerSetRowsStateless, so this pass claims the KV-cache
/// SetRows ops and the default stateless lowering only sees the rest (e.g. MoE routing writes).
///
/// Per KV cache it replaces the Parameter/Result pair with a Variable + ReadValue(empty init) +
/// Gather(beam_idx) + Concat(past, this step's rows) + Assign. Only a SetRows writing to a model
/// Parameter is converted.
///
/// Two details that are load-bearing rather than cosmetic:
///  - `beam_idx` is ADDED here, not taken from the decoder: ggml has no counterpart, so declaring
///    it in a decoder would leave a consumer-less input on the stateless graph.
///  - the ReadValue init must be empty: CPU's stateful_sdpa_fusion folds the cache into
///    ScaledDotProductAttentionWithKVCache, whose MemoryInputSDPA aborts on zero parent edges.
///
/// Scope: this grows the cache and deliberately does not touch the attention mask. The native
/// builder emits a dynamically sized mask, which needs no change; a graph that preallocates a
/// fixed mask window must be re-sliced by the caller, as the llama.cpp backend does.
/// Key under which the frontend records the model's recurrent (overwritten, non-appending) states
/// in rt_info, as a flat list of alternating {input name, output name}. Linear-attention
/// architectures (qwen35's Gated DeltaNet) carry a conv window and a delta matrix per recurrent
/// layer. Unlike a KV cache these have no token axis and no SetRows write marking them in the
/// graph, so the pairing has to be carried explicitly; see GgufDecoder::get_recurrent_states.
GGUF_FRONTEND_API const std::string& gguf_recurrent_states_key();

/// LIMITATION -- recurrent states are batch-1 and are NOT reordered by beam_idx. Unlike a KV
/// cache, which this pass gathers by beam_idx before appending, a recurrent state is a single
/// static-shaped block with no batch axis to reorder. Beam search or batch > 1 therefore fails at
/// inference with a shape mismatch on the state's Concat rather than silently mixing state across
/// beams; greedy, batch-1 generation is the supported mode for a linear-attention architecture.
///
/// Key under which the frontend records that the model uses interleaved M-RoPE (qwen35 /
/// qwen3vl). Such a model expects inp_pos to carry FOUR position sections per token, so a consumer
/// feeding it plain per-token positions (as OpenVINO GenAI does) has to expand them first.
GGUF_FRONTEND_API const std::string& gguf_imrope_key();

/// Key under which the frontend records the model's sliding-window length in tokens (int64_t),
/// when the GGUF metadata carries an explicit value. Recorded so AdaptToGenAI can build a
/// correctly windowed self_kq_mask_swa (rather than reusing the full causal mask) without
/// re-reading the .gguf file. Absent from rt_info when the model has no SWA, or its SWA is
/// described only by sinks / a per-layer boolean pattern with no accompanying token count.
GGUF_FRONTEND_API const std::string& gguf_swa_window_key();

class GGUF_FRONTEND_API GGUFMakeStateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("gguf::GGUFMakeStateful");

    /// \param skip_caches Friendly names of cache Parameters to leave stateless. A sliding-window
    ///        cache is evicted from the front rather than only appended to, so an append-grown
    ///        Variable would not reproduce it.
    /// \param append_axis Cache axis the new rows are appended along (the token axis). -1 infers
    ///        it as the cache Parameter's single dynamic axis. Pass an explicit axis for a fully
    ///        static (preallocated) cache, where there is nothing to infer from.
    /// \param beam_idx_name Name of the beam-reorder input, which this pass ADDS to the model (no
    ///        decoder declares it, since it indexes OpenVINO state that ggml has no counterpart
    ///        for). The past cache is gathered by it along the batch axis before the append
    ///        Concat; with batch 1 that Gather is an identity, but emitting it is what lets CPU's
    ///        stateful_sdpa_fusion match and makes beam search work. A model that already carries
    ///        a Parameter of this name has it reused instead.
    explicit GGUFMakeStateful(std::set<std::string> skip_caches = {},
                              int64_t append_axis = -1,
                              std::string beam_idx_name = "beam_idx")
        : m_skip_caches(std::move(skip_caches)),
          m_append_axis(append_axis),
          m_beam_idx_name(std::move(beam_idx_name)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::set<std::string> m_skip_caches;
    int64_t m_append_axis;
    std::string m_beam_idx_name;
};

}  // namespace ov::frontend::gguf::pass
