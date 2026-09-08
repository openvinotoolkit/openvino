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

    // Invoke a registered converter; its OpenVINO outputs supply shape/type inference.
    // out_type is the GGML result type requested by the operation, not a shape-inference hint.
    GgufValue node(const std::string& op_type,
                   const std::vector<GgufValue>& inputs,
                   ov::element::Type out_type,
                   int op_case = 0,
                   const std::map<std::string, ov::Any>& attrs = {});

    // Declare the position contract before emitting nodes; config.per_op selects per-node tables.
    void configure_rope(const RopeConfig& config);

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
