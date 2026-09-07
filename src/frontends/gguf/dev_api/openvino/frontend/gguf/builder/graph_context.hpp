// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/frontend/gguf/builder/decoder_options.hpp"
#include "openvino/frontend/gguf/builder/model_builder.hpp"
#include "openvino/frontend/gguf/builder/tensor_table.hpp"
#include "openvino/frontend/gguf/builder/value.hpp"
#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Generic graph operations and adapters to the frontend's shared decoder blocks. This API uses
// GGML operand order and reversed dimension order; it is not a clone of llama.cpp's graph API.
// Built-in and external architectures use the same metadata resolver and decoder blocks.
class GGUF_FRONTEND_API GgufGraphContext {
public:
    explicit GgufGraphContext(const BuildContext& ctx);
    ~GgufGraphContext();

    GgufGraphContext(const GgufGraphContext&) = delete;
    GgufGraphContext& operator=(const GgufGraphContext&) = delete;

    // ---- what is being built ----
    const GgufMetadata& metadata() const;
    // Resolve decoder metadata once, before building decoder blocks. Non-decoder families never
    // need this. Options are validated before derived plans and model metadata are computed.
    DecoderDimensions configure_decoder(RopeMode rope, const DecoderOptions& options = {});
    DecoderLayerParameters decoder_layer_parameters(int layer) const;
    GgufValue decoder_attention(int layer, const GgufValue& normalized_input);
    GgufValue decoder_ffn(int layer, const GgufValue& normalized_input);
    const std::string& arch() const;
    GgufTensors tensors();

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
    // A -1 denotes an inferred dimension. No attention layout is inferred from the target rank.
    GgufValue reshape(const GgufValue& x, const std::vector<int64_t>& ne);
    // Explicit decoder layout operations preserve a dynamic token axis and the leading batch.
    GgufValue split_heads(const GgufValue& x, int64_t heads, int64_t head_size);
    GgufValue merge_heads(const GgufValue& x);
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

    // ---- normalization blocks ----

    // RMS norm, optionally scaled by `w` (pass an empty value for llama.cpp's NULL weight, which
    // means a plain normalization with no multiplicative term).
    GgufValue build_norm(const GgufValue& cur, const GgufValue& w, float eps);
    // LayerNorm with optional weight and bias.
    GgufValue build_norm_ln(const GgufValue& cur, const GgufValue& w, const GgufValue& b, float eps);

    // ---- finishing ----
    // Mark `logits` as the model's output. The port of `res->t_logits = cur` plus
    // ggml_build_forward_expand.
    void set_output(const GgufValue& logits);
    // Model contracts used by the existing normalization passes.
    void set_sliding_window(int64_t tokens);
    // Registers an overwritten state, automatically marking the update as a model output.
    void add_recurrent_state(const GgufValue& input, const GgufValue& update);
    // Validate and seal the finished graph. Call once, last.
    std::shared_ptr<GgufGraph> finish();

private:
    friend class GgufTensors;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
