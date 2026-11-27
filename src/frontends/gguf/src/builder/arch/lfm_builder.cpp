// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lfm_builder.hpp"

#include <algorithm>
#include <limits>

#include "builder/decoder_metadata.hpp"

namespace ov::frontend::gguf {
namespace {
// Noncausal LFM2 encoders and LFM3 D1 read a whole prompt in one stateless batch.
void make_stateless(DecoderParameters& parameters) {
    parameters.embedding_model = true;
    parameters.model_contract.batchable_recurrent.reset();
    parameters.model_contract.sequence_outputs.clear();
    parameters.model_contract.supports_genai_adaptation = false;
}

DecoderParameters lfm_decoder_parameters(const GgufMetadata& metadata, const std::string& arch) {
    const bool lfm3 = arch == "lfm3";
    if (!lfm3) {
        DecoderOptions options;
        options.normalize_expert_weights = true;
        auto parameters = resolve_decoder_parameters(metadata, RopeMode::Neox, options);
        if (!parameters.causal_attention) {
            OPENVINO_ASSERT(arch == "lfm2", "[GGUF] noncausal attention is supported only for lfm2, not ", arch);
            OPENVINO_ASSERT(parameters.attention.window_size <= 0,
                            "[GGUF] noncausal LFM2 does not support sliding-window attention");
            OPENVINO_ASSERT(metadata.get_int("lfm2.decision.block_count").value_or(0) == 0,
                            "[GGUF] LFM2 decision heads are not supported");
            make_stateless(parameters);
        }
        return parameters;
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
    if (!parameters.causal_attention) {
        make_stateless(parameters);
        parameters.pooling_type = 0;
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
        const auto parameters = lfm_decoder_parameters(g.metadata(), g.arch());
        const bool causal = parameters.causal_attention;
        // Noncausal LFM2 is a sentence or token encoder; noncausal LFM3 (D1) keeps its LM head.
        const bool encoder = !causal && !lfm3;
        const auto dims = g.configure_decoder(parameters);
        OPENVINO_ASSERT(dims.layers > 0 && dims.embedding > 0,
                        "[GGUF] LFM decoder requires positive layer and embedding counts");
        auto weights = g.tensors();
        const bool detokenizer = causal && !lfm3 && weights.has("dense_2.weight");
        const auto output_weight = encoder                        ? weights("dense_2.weight")
                                   : detokenizer                  ? weights.require("dense_2.weight")
                                   : weights.has("output.weight") ? weights.require("output.weight")
                                                                  : weights.require("token_embd.weight");
        if (detokenizer)
            OPENVINO_ASSERT(output_weight.ne(0) == dims.embedding && output_weight.ne(1) == 1282,
                            "[GGUF] LFM2 audio detokenizer requires a 1282-channel spectral head");
        g.build_inp_pos();
        if (causal)
            g.build_attn_inp_kv();
        else if (lfm3)
            g.add_input("self_kq_mask_swa", ov::element::f32, ov::PartialShape{1, 1, -1, -1});
        auto cur = g.build_inp_embd(weights.require("token_embd.weight"));
        const auto out_ids = detokenizer || encoder ? GgufValue{} : g.build_inp_out_ids();
        const auto conv_padding = causal ? LfmConvPadding::STATE
                                  : lfm3 ? LfmConvPadding::CAUSAL
                                         : LfmConvPadding::CENTERED;
        OPENVINO_ASSERT(conv_padding != LfmConvPadding::CENTERED || *kernel % 2 == 1,
                        "[GGUF] a centered short convolution requires an odd kernel");
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
                OPENVINO_ASSERT(conv.ne(0) == *kernel,
                                "[GGUF] short-convolution weight width does not match shortconv.l_cache in layer ",
                                layer);
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
                                                        "conv_state_l" + std::to_string(layer),
                                                        true,
                                                        conv_padding);
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
        if (encoder) {
            g.set_output(encoder_output(cur, output_weight, parameters.pooling_type), "embeddings");
            return g.finish();
        }
        if (lfm3 && causal)
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
    // Pools [1, 1, T, D] like llama.cpp build_pooling, then applies the optional dense_2 projection.
    GgufValue encoder_output(GgufValue cur, const GgufValue& dense, int pooling) {
        const auto tokens = [&](int64_t start) {
            return g.node("GGML_OP_VIEW", {cur}, 3, {{"view_slice", std::vector<int64_t>{2, start, 1}}});
        };
        if (pooling == 1)
            cur = g.node("GGML_OP_TRANSPOSE", {g.node("GGML_OP_MEAN", {g.node("GGML_OP_TRANSPOSE", {cur})})});
        else if (pooling == 2)
            cur = tokens(0);
        else if (pooling == 3)
            cur = tokens(-1);
        auto width = cur.ne(0);
        if (dense) {
            width = dense.ne(1);
            OPENVINO_ASSERT(
                dense.ne(0) == cur.ne(0) && g.metadata().get_int("lfm2.embedding_length_out").value_or(width) == width,
                "[GGUF] LFM2 dense_2 must project the hidden size to embedding_length_out");
            cur = g.node("GGML_OP_MUL_MAT", {dense, cur});
            if (auto bias = g.tensors()("dense_2.bias")) {
                OPENVINO_ASSERT(bias.ne(0) == width, "[GGUF] LFM2 dense_2.bias must match embedding_length_out");
                cur = g.node("GGML_OP_ADD", {cur, bias});
            }
        }
        return g.node("GGML_OP_RESHAPE", {cur}, 0, {{"reshape_target", std::vector<int64_t>{-1, width}}});
    }

    GgufGraphContext g;
};
}  // namespace

GgufValue build_lfm_short_conv(GgufGraphContext& g,
                               const GgufValue& projected,
                               const GgufValue& conv_weight,
                               int64_t channels,
                               const std::string& state_name,
                               bool batch_major,
                               LfmConvPadding padding) {
    const auto slice = [&](const GgufValue& value, int64_t start, int64_t length) {
        return g.node("GGML_OP_VIEW", {value}, 3, {{"view_slice", std::vector<int64_t>{3, start, length}}});
    };
    const auto taps = conv_weight.ne(0) - 1;
    const auto bx =
        g.node("GGML_OP_TRANSPOSE",
               {g.node("GGML_OP_MUL", {slice(projected, 0, channels), slice(projected, 2 * channels, channels)})});
    const auto zeros = [&](const std::string& suffix, int64_t length) {
        ov::Tensor value(ov::element::f32, {1, 1, static_cast<size_t>(channels), static_cast<size_t>(length)});
        std::fill_n(value.data<float>(), value.get_size(), 0.0f);
        return g.add_constant(state_name + suffix, value);
    };
    const auto concat = [&](const GgufValue& left, const GgufValue& right) {
        return g.node("GGML_OP_CONCAT", {left, right}, 0, {{"concat_axis", 0}});
    };
    GgufValue window;
    if (padding == LfmConvPadding::STATE) {
        const auto state = g.add_input(state_name, ov::element::f32, {1, 1, channels, taps});
        window = concat(state, bx);
        g.add_recurrent_state(state, slice(window, -taps, taps));
    } else if (padding == LfmConvPadding::CAUSAL) {
        window = concat(zeros(".pad", taps), bx);
    } else {
        OPENVINO_ASSERT(taps % 2 == 0, "[GGUF] a centered short convolution requires an odd kernel");
        const auto pad = zeros(".pad", taps / 2);
        window = concat(concat(pad, bx), pad);
    }
    const auto conv = g.node("GGML_OP_SSM_CONV", {window, conv_weight}, 0, {{"batch_major", batch_major}});
    return g.node("GGML_OP_MUL", {slice(projected, channels, channels), conv});
}
std::shared_ptr<ModelBuilder> make_lfm_builder(const BuildContext& context) {
    return std::make_shared<LfmBuilder>(context);
}
}  // namespace ov::frontend::gguf
