// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lfm_builder.hpp"

namespace ov::frontend::gguf {
std::shared_ptr<GgufGraph> build_lfm_decoder(GgufGraphContext& g,
                                             const DecoderDimensions& dims,
                                             const LfmDecoderConfig& config) {
    OPENVINO_ASSERT(config.layers.size() == static_cast<size_t>(dims.layers),
                    "[GGUF] LFM decoder requires a configuration for every layer");
    g.build_inp_pos();
    g.build_attn_inp_kv();
    auto weights = g.tensors();
    auto cur = g.build_inp_embd(weights.require("token_embd.weight"));
    const auto out_ids = config.select_output_rows ? g.build_inp_out_ids() : GgufValue{};
    for (int layer = 0; layer < dims.layers; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        const auto weight = [&](const std::string& name) {
            return weights.require(prefix + name);
        };
        const auto& block = config.layers[layer];
        auto residual = cur;
        cur = g.build_norm(cur, weight("attn_norm.weight"), dims.norm_epsilon);
        if (block.recurrent) {
            const auto gated =
                build_lfm_short_conv(g,
                                     g.node("GGML_OP_MUL_MAT", {weight("shortconv.in_proj.weight"), cur}),
                                     weight("shortconv.conv.weight"),
                                     dims.embedding,
                                     "conv_state_l" + std::to_string(layer));
            cur = g.node("GGML_OP_MUL_MAT", {weight("shortconv.out_proj.weight"), gated});
        } else {
            cur = g.decoder_attention(layer, cur);
        }
        if (block.sandwich_norm)
            cur = g.build_norm(cur, weight("post_attention_norm.weight"), dims.norm_epsilon);
        if (layer == dims.layers - 1 && out_ids) {
            cur = g.node("GGML_OP_GET_ROWS", {cur, out_ids});
            residual = g.node("GGML_OP_GET_ROWS", {residual, out_ids});
        }
        cur = g.node("GGML_OP_ADD", {residual, cur});
        residual = cur;
        cur = g.decoder_ffn(layer, g.build_norm(cur, weight("ffn_norm.weight"), dims.norm_epsilon));
        if (block.sandwich_norm)
            cur = g.build_norm(cur, weight("post_ffw_norm.weight"), dims.norm_epsilon);
        cur = g.node("GGML_OP_ADD", {residual, cur});
    }
    cur = g.build_norm(cur, config.output_norm, dims.norm_epsilon);
    if (!config.hidden_output.empty())
        g.set_output(cur, config.hidden_output);
    auto output = g.node("GGML_OP_MUL_MAT", {config.output_weight, cur});
    if (config.output_bias)
        output = g.node("GGML_OP_ADD", {output, config.output_bias});
    g.set_primary_output(output);
    return g.finish();
}

GgufValue build_lfm_short_conv(GgufGraphContext& g,
                               const GgufValue& projected,
                               const GgufValue& conv_weight,
                               int64_t channels,
                               const std::string& state_name,
                               bool batch_major) {
    const auto slice = [&](const GgufValue& value, int64_t start, int64_t length) {
        return g.node("GGML_OP_VIEW", {value}, 3, {{"view_slice", std::vector<int64_t>{3, start, length}}});
    };
    const auto taps = conv_weight.ne(0) - 1;
    const auto bx =
        g.node("GGML_OP_TRANSPOSE",
               {g.node("GGML_OP_MUL", {slice(projected, 0, channels), slice(projected, 2 * channels, channels)})});
    const auto state = g.add_input(state_name, ov::element::f32, {1, 1, channels, taps});
    const auto window = g.node("GGML_OP_CONCAT", {state, bx}, 0, {{"concat_axis", 0}});
    g.add_recurrent_state(state, slice(window, -taps, taps));
    const auto conv = g.node("GGML_OP_SSM_CONV", {window, conv_weight}, 0, {{"batch_major", batch_major}});
    return g.node("GGML_OP_MUL", {slice(projected, channels, channels), conv});
}
}  // namespace ov::frontend::gguf
