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

// Graph operations and shared decoder blocks. Operands use GGML order; shapes use OpenVINO order.
class GGUF_FRONTEND_API GgufGraphContext {
public:
    explicit GgufGraphContext(const BuildContext& ctx);
    ~GgufGraphContext();

    GgufGraphContext(const GgufGraphContext&) = delete;
    GgufGraphContext& operator=(const GgufGraphContext&) = delete;

    // Model configuration
    const GgufMetadata& metadata() const;
    // Call once before using decoder blocks; optional for other families.
    // Options are validated before deriving execution plans and graph metadata.
    DecoderDimensions configure_decoder(RopeMode rope, const DecoderOptions& options = {});
    DecoderLayerParameters decoder_layer_parameters(int layer) const;
    GgufValue decoder_attention(int layer, const GgufValue& normalized_input);
    GgufValue decoder_ffn(int layer, const GgufValue& normalized_input);
    const std::string& arch() const;
    GgufTensors tensors();

    // Inputs
    // GET_ROWS(tok_embd, inp_tokens), creating inp_tokens on first use.
    GgufValue build_inp_embd(const GgufValue& tok_embd);
    // Position indices consumed by RoPE.
    GgufValue build_inp_pos();
    // Row selector for the last layer's output (inp_out_ids).
    GgufValue build_inp_out_ids();
    // Declare masks, KV write indices, and token lengths before the layer loop.
    // The SWA mask is included when requested here or by configure_decoder.
    void build_attn_inp_kv(bool swa = false);
    GgufValue add_input(const std::string& name, ov::element::Type type, const ov::PartialShape& shape);

    // Operations with inferred output shapes
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
    // Explicit shape in GGML order (fastest axis first), with at most one inferred dimension (-1).
    // Use split_heads/merge_heads for attention layouts.
    GgufValue reshape(const GgufValue& x, const std::vector<int64_t>& ne);
    // Explicit decoder layout operations preserve a dynamic token axis and the leading batch.
    GgufValue split_heads(const GgufValue& x, int64_t heads, int64_t head_size);
    GgufValue merge_heads(const GgufValue& x);
    GgufValue cont(const GgufValue& x);
    GgufValue transpose(const GgufValue& x);
    // Permutation of OpenVINO axes [0, 1, 2, 3]; axis 3 is GGML ne[0].
    GgufValue permute(const GgufValue& x, const std::vector<int64_t>& perm);
    // Concatenate along a ggml dimension index.
    GgufValue concat(const GgufValue& a, const GgufValue& b, int ggml_dim);
    // RoPE. `positions` is normally build_inp_pos(); `freq_factors` may be empty.
    GgufValue rope_ext(const GgufValue& x,
                       const GgufValue& positions,
                       const GgufValue& freq_factors,
                       const RopeConfig& cfg,
                       int rope_op_case);

    // Emit a GGML operation with explicit output metadata and translator-specific attributes.
    GgufValue raw_op(const std::string& op_type,
                     const std::vector<GgufValue>& inputs,
                     const ov::PartialShape& out_shape,
                     ov::element::Type out_type,
                     int op_case = 0,
                     const std::map<std::string, ov::Any>& attrs = {});

    // Normalization

    // RMS norm with optional weight scaling; an empty w leaves it unscaled.
    GgufValue build_norm(const GgufValue& cur, const GgufValue& w, float eps);
    // LayerNorm with optional weight and bias.
    GgufValue build_norm_ln(const GgufValue& cur, const GgufValue& w, const GgufValue& b, float eps);

    // Outputs and state
    void set_output(const GgufValue& logits);
    // Record the sliding window for normalization passes.
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
