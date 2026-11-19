// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lfm_builder.hpp"

#include <algorithm>
#include <limits>

#include "builder/decoder_metadata.hpp"

namespace ov::frontend::gguf {
namespace {
DecoderParameters lfm_decoder_parameters(const GgufMetadata& metadata, bool lfm3) {
    if (!lfm3) {
        DecoderOptions options;
        options.normalize_expert_weights = true;
        return resolve_decoder_parameters(metadata, RopeMode::Neox, options);
    }
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
    return parameters;
}

class LfmBuilder : public ModelBuilder {
public:
    explicit LfmBuilder(const BuildContext& context) : g(context) {}

    std::shared_ptr<GgufGraph> build() override {
        const bool lfm3 = g.arch() == "lfm3";
        const auto kernel = g.metadata().get_int(g.arch() + ".shortconv.l_cache");
        OPENVINO_ASSERT(kernel && *kernel > 1 && *kernel <= std::numeric_limits<int>::max(),
                        "[GGUF] ",
                        lfm3 ? "LFM3" : "LFM2",
                        " requires shortconv.l_cache > 1");
        OPENVINO_ASSERT(g.metadata().get_bool(g.arch() + ".attention.causal").value_or(true),
                        lfm3 ? "[GGUF] LFM3 decision models are not supported"
                             : "[GGUF] noncausal LFM2 embedding models require a centered convolution builder");
        const auto parameters = lfm_decoder_parameters(g.metadata(), lfm3);
        const auto dims = g.configure_decoder(parameters);
        OPENVINO_ASSERT(dims.layers > 0 && dims.embedding > 0,
                        "[GGUF] LFM decoder requires positive layer and embedding counts");
        auto weights = g.tensors();
        const bool detokenizer = !lfm3 && weights.has("dense_2.weight");
        const auto output_weight = weights.require(detokenizer                    ? "dense_2.weight"
                                                   : weights.has("output.weight") ? "output.weight"
                                                                                  : "token_embd.weight");
        if (detokenizer)
            OPENVINO_ASSERT(output_weight.ne(0) == dims.embedding && output_weight.ne(1) == 1282,
                            "[GGUF] LFM2 audio detokenizer requires a 1282-channel spectral head");
        g.build_inp_pos();
        g.build_attn_inp_kv();
        auto cur = g.build_inp_embd(weights.require("token_embd.weight"));
        const auto out_ids = detokenizer ? GgufValue{} : g.build_inp_out_ids();
        for (int layer = 0; layer < dims.layers; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto weight = [&](const std::string& name) {
                return weights.require(prefix + name);
            };
            const bool post_norm = lfm3 && weights.has(prefix + "post_attention_norm.weight");
            if (lfm3)
                OPENVINO_ASSERT(post_norm == weights.has(prefix + "post_ffw_norm.weight"),
                                "[GGUF] LFM3 requires both post norms or neither in layer ",
                                layer);
            const bool recurrent = lfm3 ? parameters.dimensions.kv_heads_per_layer[layer] == 0
                                        : weights.has(prefix + "shortconv.in_proj.weight");
            auto residual = cur;
            cur = g.build_norm(cur, weight("attn_norm.weight"), dims.norm_epsilon);
            if (recurrent) {
                const auto in = weight("shortconv.in_proj.weight");
                const auto conv = weight("shortconv.conv.weight");
                const auto out = weight("shortconv.out_proj.weight");
                if (lfm3)
                    OPENVINO_ASSERT(in.ne(0) == dims.embedding && in.ne(1) == int64_t{3} * dims.embedding &&
                                        conv.ne(0) == *kernel && conv.ne(1) == dims.embedding &&
                                        out.ne(0) == dims.embedding && out.ne(1) == dims.embedding,
                                    "[GGUF] invalid LFM3 short-convolution weights in layer ",
                                    layer);
                const auto gated = build_lfm_short_conv(g,
                                                        g.node("GGML_OP_MUL_MAT", {in, cur}),
                                                        conv,
                                                        dims.embedding,
                                                        "conv_state_l" + std::to_string(layer));
                cur = g.node("GGML_OP_MUL_MAT", {out, gated});
            } else {
                if (lfm3) {
                    OPENVINO_ASSERT(!weights.has(prefix + "shortconv.in_proj.weight"),
                                    "[GGUF] LFM3 convolution tensor conflicts with attention-layer metadata");
                    OPENVINO_ASSERT(weight("attn_gate.weight").ne(1) == g.decoder_layer_parameters(layer).query_heads,
                                    "[GGUF] LFM3 requires a sigmoid gate per attention head");
                    if (!parameters.attention.sliding_layers[layer])
                        weights.require("rope_freqs.weight");
                }
                cur = g.decoder_attention(layer, cur);
            }
            if (post_norm)
                cur = g.build_norm(cur, weight("post_attention_norm.weight"), dims.norm_epsilon);
            if (layer == dims.layers - 1 && out_ids) {
                cur = g.node("GGML_OP_GET_ROWS", {cur, out_ids});
                residual = g.node("GGML_OP_GET_ROWS", {residual, out_ids});
            }
            cur = g.node("GGML_OP_ADD", {residual, cur});
            residual = cur;
            cur = g.decoder_ffn(layer, g.build_norm(cur, weight("ffn_norm.weight"), dims.norm_epsilon));
            if (post_norm)
                cur = g.build_norm(cur, weight("post_ffw_norm.weight"), dims.norm_epsilon);
            cur = g.node("GGML_OP_ADD", {residual, cur});
        }
        cur = g.build_norm(cur,
                           weights.require(lfm3 ? "output_norm.weight" : "token_embd_norm.weight"),
                           dims.norm_epsilon);
        if (lfm3)
            g.set_output(cur, "audio_hidden_states");
        auto output = g.node("GGML_OP_MUL_MAT", {output_weight, cur});
        if (detokenizer) {
            if (auto bias = weights("dense_2.bias"))
                output = g.node("GGML_OP_ADD", {output, bias});
        }
        g.set_primary_output(output);
        return g.finish();
    }

private:
    GgufGraphContext g;
};
}  // namespace

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
std::shared_ptr<ModelBuilder> make_lfm_builder(const BuildContext& context) {
    return std::make_shared<LfmBuilder>(context);
}
}  // namespace ov::frontend::gguf
