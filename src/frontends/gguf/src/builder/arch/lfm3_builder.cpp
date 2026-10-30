// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lfm3_builder.hpp"

#include <algorithm>
#include <limits>

#include "lfm2_builder.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov::frontend::gguf {
namespace {
class Lfm3Builder : public ModelBuilder {
public:
    explicit Lfm3Builder(const BuildContext& context) : g(context) {}

    std::shared_ptr<GgufGraph> build() override {
        const auto& metadata = g.metadata();
        const auto kernel = metadata.get_int("lfm3.shortconv.l_cache");
        OPENVINO_ASSERT(kernel && *kernel > 1 && *kernel <= std::numeric_limits<int>::max(),
                        "[GGUF] LFM3 requires shortconv.l_cache > 1");
        OPENVINO_ASSERT(metadata.get_bool("lfm3.attention.causal").value_or(true),
                        "[GGUF] LFM3 decision models are not supported");
        OPENVINO_ASSERT(metadata.has("lfm3.attention.sliding_window_pattern") &&
                            metadata.has("lfm3.attention.sliding_window") && metadata.has("lfm3.rope.freq_base_swa"),
                        "[GGUF] LFM3 requires sliding-window metadata and its RoPE frequency base");
        DecoderParameters defaults;
        defaults.attention.rope_mode = RopeMode::Neox;
        defaults.attention.output_gate = true;
        defaults.normalization.qk_norm = true;
        defaults.normalization.qk_full_width = false;
        defaults.model_contract.batchable_recurrent = true;
        defaults.model_contract.sequence_outputs = {"audio_hidden_states"};
        auto parameters = read_decoder_parameters(metadata, defaults);
        parameters.experts.enabled = false;
        const auto layers = static_cast<size_t>(parameters.dimensions.layers);
        const auto& kv_heads = parameters.dimensions.kv_heads_per_layer;
        const auto& sliding = parameters.attention.sliding_layers;
        OPENVINO_ASSERT(kv_heads.size() == layers && sliding.size() == layers && parameters.attention.window_size > 0 &&
                            std::any_of(sliding.begin(),
                                        sliding.end(),
                                        [](int value) {
                                            return value == 1;
                                        }),
                        "[GGUF] LFM3 requires per-layer KV heads and sliding-window flags");
        const auto value_size = metadata.get_int("lfm3.attention.value_length");
        OPENVINO_ASSERT(!value_size || *value_size == parameters.dimensions.head_size,
                        "[GGUF] LFM3 requires equal key and value head sizes");
        parameters.layers.resize(layers);
        for (size_t layer = 0; layer < layers; ++layer) {
            OPENVINO_ASSERT((sliding[layer] == 0 || sliding[layer] == 1) && (!sliding[layer] || kv_heads[layer] > 0),
                            "[GGUF] LFM3 sliding-window flags must select attention layers");
            parameters.layers[layer].recurrent = kv_heads[layer] == 0;
        }
        const auto dims = g.configure_decoder(parameters);
        g.build_inp_pos();
        g.build_attn_inp_kv();
        auto weights = g.tensors();
        auto cur = g.build_inp_embd(weights.require("token_embd.weight"));
        const auto out_ids = g.build_inp_out_ids();
        for (int layer = 0; layer < dims.layers; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto weight = [&](const std::string& name) {
                return weights.require(prefix + name);
            };
            const bool post_attention = weights.has(prefix + "post_attention_norm.weight");
            OPENVINO_ASSERT(post_attention == weights.has(prefix + "post_ffw_norm.weight"),
                            "[GGUF] LFM3 requires both post norms or neither in layer ",
                            layer);
            auto residual = cur;
            cur = g.build_norm(cur, weight("attn_norm.weight"), dims.norm_epsilon);
            if (kv_heads[layer] == 0) {
                const auto in = weight("shortconv.in_proj.weight");
                const auto convolution_weight = weight("shortconv.conv.weight");
                const auto out = weight("shortconv.out_proj.weight");
                OPENVINO_ASSERT(in.ne(0) == dims.embedding && in.ne(1) == int64_t{3} * dims.embedding &&
                                    convolution_weight.ne(0) == *kernel && convolution_weight.ne(1) == dims.embedding &&
                                    out.ne(0) == dims.embedding && out.ne(1) == dims.embedding,
                                "[GGUF] invalid LFM3 short-convolution weights in layer ",
                                layer);
                const auto gated = build_lfm_short_conv(g,
                                                        g.node("GGML_OP_MUL_MAT", {in, cur}),
                                                        convolution_weight,
                                                        dims.embedding,
                                                        "conv_state_l" + std::to_string(layer));
                cur = g.node("GGML_OP_MUL_MAT", {out, gated});
            } else {
                OPENVINO_ASSERT(!weights.has(prefix + "shortconv.in_proj.weight"),
                                "[GGUF] LFM3 convolution tensor conflicts with attention-layer metadata");
                const auto head = g.decoder_layer_parameters(layer);
                OPENVINO_ASSERT(weight("attn_gate.weight").ne(1) == head.query_heads,
                                "[GGUF] LFM3 requires a sigmoid gate per attention head");
                if (!sliding[layer])
                    weights.require("rope_freqs.weight");
                cur = g.decoder_attention(layer, cur);
            }
            if (post_attention)
                cur = g.build_norm(cur, weight("post_attention_norm.weight"), dims.norm_epsilon);
            if (layer == dims.layers - 1) {
                cur = g.node("GGML_OP_GET_ROWS", {cur, out_ids});
                residual = g.node("GGML_OP_GET_ROWS", {residual, out_ids});
            }
            cur = g.node("GGML_OP_ADD", {residual, cur});
            residual = cur;
            cur = g.decoder_ffn(layer, g.build_norm(cur, weight("ffn_norm.weight"), dims.norm_epsilon));
            if (post_attention)
                cur = g.build_norm(cur, weight("post_ffw_norm.weight"), dims.norm_epsilon);
            cur = g.node("GGML_OP_ADD", {residual, cur});
        }
        cur = g.build_norm(cur, weights.require("output_norm.weight"), dims.norm_epsilon);
        g.set_output(cur, "audio_hidden_states");
        const auto output =
            weights.has("output.weight") ? weights.require("output.weight") : weights.require("token_embd.weight");
        g.set_primary_output(g.node("GGML_OP_MUL_MAT", {output, cur}));
        return g.finish();
    }

private:
    GgufGraphContext g;
};
}  // namespace

std::shared_ptr<ModelBuilder> make_lfm3_builder(const BuildContext& context) {
    return std::make_shared<Lfm3Builder>(context);
}
}  // namespace ov::frontend::gguf
