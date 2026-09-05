// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/frontend/gguf/builder/hparams.hpp"
#include "openvino/frontend/gguf/builder/model_builder.hpp"
#include "openvino/frontend/gguf/builder/tensor_table.hpp"
#include "openvino/frontend/gguf/builder/value.hpp"
#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Activation of a gated feed-forward block, matching llama.cpp's llm_ffn_op_type.
enum class FfnOp {
    Silu,  // SwiGLU (llama, qwen, phi3, ...)
    Gelu,  // GeGLU  (gemma)
    Relu,
};

// Optional inputs of an attention block that only some architectures carry.
struct GGUF_FRONTEND_API AttnOptions {
    // Per-head sink logit, the 5th FLASH_ATTN_EXT input (gpt-oss).
    GgufValue sinks;
    // Which mask input to attend against. Sliding-window layers use "self_kq_mask_swa".
    std::string mask = "self_kq_mask";
    // Attention logit soft-cap (gemma2); 0 disables.
    float kq_soft_cap = 0.0f;
};

// The graph-building vocabulary an extension writes a model against: the port target for a
// llama.cpp `src/models/<arch>.cpp` file, and the counterpart of its `llm_graph_context`.
//
// Two levels are available, and a port normally uses both:
//
//   * the ggml op vocabulary (add, mul_mat, rope_ext, reshape, ...), which mirrors the `ggml_*`
//     calls a model file makes directly. Unlike raw ggml these INFER each op's output shape from
//     its inputs, which is what removes the shape bookkeeping that would otherwise dominate a port.
//
//   * the `build_*` blocks (build_norm, build_ffn, build_attn, ...), which mirror the
//     llm_graph_context methods and expand to the same multi-op subgraphs llama.cpp's do.
//
// Anything not covered by either is reachable through raw_op(), which appends a node in the GGML
// op vocabulary directly. Pair it with an ov::frontend::ConversionExtension when the op is one the
// frontend does not yet translate: the two extension types compose, so a genuinely new operation
// does not require a frontend change either.
//
// Shapes are carried in the OpenVINO/GGML logical order [ne3, ne2, ne1, ne0], the reverse of
// ggml's ne[] indexing; GgufValue::ne() reads them back in ggml order.
class GGUF_FRONTEND_API GgufGraphContext {
public:
    explicit GgufGraphContext(const BuildContext& ctx);
    ~GgufGraphContext();

    GgufGraphContext(const GgufGraphContext&) = delete;
    GgufGraphContext& operator=(const GgufGraphContext&) = delete;

    // ---- what is being built ----
    const GgufMetadata& metadata() const;
    const GgufHparams& hparams() const;
    const std::string& arch() const;
    GgufTensors tensors();

    // Representative token count used for per-node output shapes.
    //
    // Per-node shapes are STATIC, as in the llama.cpp cgraph path, which builds its graph for one
    // concrete token length. The real dynamic-ness lives in the model's input Parameters; this
    // value only feeds each node's shape metadata, which translators consult. A port that would
    // write `n_tokens` in llama.cpp writes this.
    int64_t n_tokens() const;

    // ---- model inputs ----
    // Token embedding lookup: GET_ROWS(tok_embd, inp_tokens). Creates "inp_tokens" on first use.
    GgufValue build_inp_embd(const GgufValue& tok_embd);
    // Position indices consumed by RoPE.
    GgufValue build_inp_pos();
    // Row selector applied to the last layer's output (llama.cpp's inp_out_ids).
    GgufValue build_inp_out_ids();
    // Declare the attention inputs: the causal mask, the KV write index, and -- when `swa` --
    // the sliding-window mask. Call once before the layer loop, like llama.cpp's
    // build_attn_inp_kv().
    void build_attn_inp_kv(bool swa = false);
    // A model input this family defines itself (a vision encoder's pixel input, say).
    GgufValue add_input(const std::string& name, ov::element::Type type, const ov::PartialShape& shape);

    // ---- ggml op vocabulary (output shapes inferred) ----
    GgufValue add(const GgufValue& a, const GgufValue& b);
    GgufValue sub(const GgufValue& a, const GgufValue& b);
    GgufValue mul(const GgufValue& a, const GgufValue& b);
    GgufValue div(const GgufValue& a, const GgufValue& b);
    GgufValue scale(const GgufValue& x, float factor);
    // Matrix multiply, ggml operand order: mul_mat(weight, activations).
    GgufValue mul_mat(const GgufValue& a, const GgufValue& b);
    GgufValue get_rows(const GgufValue& a, const GgufValue& b);
    GgufValue rms_norm(const GgufValue& x, float eps);
    GgufValue norm(const GgufValue& x, float eps);
    GgufValue soft_max(const GgufValue& x);
    GgufValue silu(const GgufValue& x);
    GgufValue gelu(const GgufValue& x);
    GgufValue gelu_quick(const GgufValue& x);
    GgufValue sigmoid(const GgufValue& x);
    GgufValue tanh(const GgufValue& x);
    GgufValue relu(const GgufValue& x);
    GgufValue sqr(const GgufValue& x);
    GgufValue sqrt(const GgufValue& x);
    // Reshape to an explicit shape, given in ggml ne order (fastest-varying dimension first),
    // so a ported `ggml_reshape_3d(ctx, cur, a, b, c)` becomes `reshape(cur, {a, b, c})`.
    GgufValue reshape(const GgufValue& x, const std::vector<int64_t>& ne);
    GgufValue cont(const GgufValue& x);
    GgufValue transpose(const GgufValue& x);
    // Reorder axes. `perm` is given in the shape's own [ne3, ne2, ne1, ne0] axis numbering (axis 3
    // is ggml's ne[0]), NOT in ggml_permute's ne order -- a ported ggml_permute therefore needs its
    // axes translated, which is the one place a port cannot be a copy.
    GgufValue permute(const GgufValue& x, const std::vector<int64_t>& perm);
    // Concatenate along a ggml dimension index.
    GgufValue concat(const GgufValue& a, const GgufValue& b, int ggml_dim);
    // RoPE. `positions` is normally build_inp_pos(); `freq_factors` may be empty.
    GgufValue rope_ext(const GgufValue& x,
                       const GgufValue& positions,
                       const GgufValue& freq_factors,
                       const RopeConfig& cfg,
                       int rope_op_case);

    // Append a node in the GGML op vocabulary directly, for anything the wrappers above do not
    // cover. The output shape and type are explicit because they cannot be inferred for an
    // arbitrary op.
    GgufValue raw_op(const std::string& op_type,
                     const std::vector<GgufValue>& inputs,
                     const ov::PartialShape& out_shape,
                     ov::element::Type out_type,
                     int op_case = 0,
                     const std::map<std::string, ov::Any>& attrs = {});

    // ---- llm_graph_context-style blocks ----

    // RMS norm, optionally scaled by `w` (pass an empty value for llama.cpp's NULL weight, which
    // means a plain normalization with no multiplicative term).
    GgufValue build_norm(const GgufValue& cur, const GgufValue& w, float eps);
    // LayerNorm with optional weight and bias.
    GgufValue build_norm_ln(const GgufValue& cur, const GgufValue& w, const GgufValue& b, float eps);

    // Gated feed-forward network. `gate` may be empty, which selects the ungated
    // (up -> activation -> down) form; each bias may be empty.
    GgufValue build_ffn(const GgufValue& cur,
                        const GgufValue& up,
                        const GgufValue& up_b,
                        const GgufValue& gate,
                        const GgufValue& gate_b,
                        const GgufValue& down,
                        const GgufValue& down_b,
                        FfnOp op);

    // Attention over an explicit Q/K/V, with the KV cache store, the attention itself and the
    // output projection -- the port of llm_graph_context::build_attn.
    //
    // Q/K/V arrive in ggml's natural [n_tokens, n_head(_kv), head_size] layout, exactly as the
    // preceding reshape/rope in a ported model file leaves them. Returns the sublayer output
    // before the residual add, as llama.cpp's build_attn does.
    GgufValue build_attn(int il,
                         const GgufValue& q,
                         const GgufValue& k,
                         const GgufValue& v,
                         const GgufValue& wo,
                         const GgufValue& wo_b,
                         float kq_scale,
                         const AttnOptions& opts = {});

    // Matrix multiply against a model weight; the port of build_lora_mm (no LoRA here, so it is
    // mul_mat, spelled the way a model file spells it).
    GgufValue build_lora_mm(const GgufValue& w, const GgufValue& cur);

    // Name a value, for readable graphs and diagnostics. The port of llama.cpp's cb(); ignoring
    // the layer index is fine, it only ever fed debug output.
    void cb(const GgufValue& v, const std::string& name, int il = -1);

    // ---- finishing ----
    // Mark `logits` as the model's output. The port of `res->t_logits = cur` plus
    // ggml_build_forward_expand.
    void set_output(const GgufValue& logits);
    // The finished graph. Call once, last.
    std::shared_ptr<GgufGraph> finish();

    // Internal: used by GgufTensors to emit a weight leaf on lookup.
    struct Impl;
    Impl& impl() {
        return *m_impl;
    }

private:
    std::unique_ptr<Impl> m_impl;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
