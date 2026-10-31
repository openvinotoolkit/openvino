// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.8-Flash-Next (llama.cpp "qwen4exp"): Gated DeltaNet and block-sparse attention (QSA) layers
// with MoE, four hyper-connection residual streams in place of every norm, and hashed n-gram
// embeddings (PLE) added to the streams before one layer. NextN (MTP) blocks are not converted.

#include "qwen4exp_builder.hpp"

#include <cmath>
#include <limits>

#include "builder/decoder_metadata.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {
namespace {

using ov::element::f32;
using ov::element::i32;

struct NgramConfig {
    int layer = -1;
    int64_t ngram_size = 0;
    int64_t heads_per_ngram = 0;
    int64_t conv_kernel = 0;
    int64_t head_dim = 0;
    int64_t eos = 0;
    std::vector<int64_t> multipliers, offsets, vocab_sizes;
};

class Qwen4ExpBuilder : public ModelBuilder {
public:
    explicit Qwen4ExpBuilder(const BuildContext& context) : g(context) {}

    std::shared_ptr<GgufGraph> build() override {
        const auto& m = g.metadata();
        const std::string a = g.arch() + ".";
        const auto required = [&](const std::string& key) {
            const auto value = m.get_int(a + key);
            OPENVINO_ASSERT(value && *value > 0, "[GGUF] qwen4exp requires a positive ", a + key);
            return *value;
        };
        hc = required("hyper_connection.count");
        OPENVINO_ASSERT(hc > 1, "[GGUF] qwen4exp requires more than one hyper-connection stream");
        rank = required("hyper_connection.low_rank");

        auto parameters = resolve_decoder_parameters(m, RopeMode::Interleaved, {});
        const auto stored_layers = static_cast<size_t>(parameters.dimensions.layers);
        auto& sparse = parameters.attention.sparse;
        const auto ratios = m.get_int_array(a + "attention.compress_ratios");
        int64_t block = 0;
        for (auto ratio : ratios) {
            OPENVINO_ASSERT(ratio >= 0 && (ratio == 0 || block == 0 || ratio == block),
                            "[GGUF] qwen4exp sparse layers must share one compress ratio");
            block = ratio ? ratio : block;
        }
        if (block > 0) {
            OPENVINO_ASSERT(ratios.size() == stored_layers, "[GGUF] qwen4exp compress ratios must cover every layer");
            sparse.key_block_size = static_cast<int>(block);
            sparse.token_budget = static_cast<int>(required("attention.indexer.top_k"));
            sparse.index_heads = static_cast<int>(required("attention.indexer.head_count"));
            sparse.index_head_size = static_cast<int>(required("attention.indexer.key_length"));
            sparse.layers.assign(ratios.begin(), ratios.end());
        }
        read_ngram(m, a, static_cast<int>(stored_layers));

        const auto dims = g.configure_decoder(parameters);
        n_embd = dims.embedding;
        eps = dims.norm_epsilon;
        weights = std::make_unique<GgufTensors>(g.tensors());
        g.build_inp_pos();
        g.build_attn_inp_kv();
        auto embeddings = g.build_inp_embd(weights->require("token_embd.weight"));
        const auto out_ids = g.build_inp_out_ids();
        GgufValue ngram_embeddings;
        if (ngram.layer >= 0)
            ngram_embeddings = build_ngram_embeddings();

        // [B, 1, T, n] -> hc identical streams [B, T, hc, n]
        auto stream = reshape(embeddings, {0, -1, 1, n_embd});
        auto streams = stream;
        for (int64_t s = 1; s < hc; ++s)
            streams = g.node("GGML_OP_CONCAT", {streams, stream}, 0, {{"concat_axis", 1}});

        for (int layer = 0; layer < dims.layers; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            if (layer == ngram.layer)
                streams = build_ple(prefix, streams, ngram_embeddings, layer);
            auto attention = mix(streams, prefix + "hc_attn_");
            auto cur = g.decoder_attention(layer, attention.mixed);
            auto inject = attention.inject;
            if (layer == dims.layers - 1) {
                cur = g.node("GGML_OP_GET_ROWS", {cur, out_ids});
                inject = g.node("GGML_OP_GET_ROWS", {inject, out_ids});
                streams = reshape(g.node("GGML_OP_GET_ROWS", {flat(streams), out_ids}), {0, -1, hc, n_embd});
            }
            streams = combine(streams, cur, inject);
            auto ffn = mix(streams, prefix + "hc_ffn_");
            streams = combine(streams, g.decoder_ffn(layer, ffn.mixed), ffn.inject);
        }
        auto head = mix(streams, "output_hc_");
        // A missing output projection is tied to the token embeddings.
        const auto output =
            weights->has("output.weight") ? weights->require("output.weight") : weights->require("token_embd.weight");
        g.set_primary_output(g.node("GGML_OP_MUL_MAT", {output, head.mixed}));
        return g.finish();
    }

private:
    struct Mix {
        GgufValue mixed;   // [B, 1, T, n]
        GgufValue inject;  // [B, 1, T, hc], empty for the output mixer
    };

    GgufValue reshape(const GgufValue& value, std::vector<int64_t> target) {
        return g.node("GGML_OP_RESHAPE", {value}, 6, {{"reshape_target", target}, {"special_zero", true}});
    }
    // [B, T, hc, n] -> [B, 1, T, hc * n]
    GgufValue flat(const GgufValue& streams) {
        return reshape(streams, {0, 1, -1, hc * n_embd});
    }
    GgufValue slice(const GgufValue& value, int64_t axis, int64_t start, int64_t length) {
        return g.node("GGML_OP_VIEW", {value}, 3, {{"view_slice", std::vector<int64_t>{axis, start, length}}});
    }
    GgufValue scale(const GgufValue& value, float factor) {
        return g.node("GGML_OP_SCALE", {value}, 0, {{"scale", factor}});
    }
    // RMS norm over each stream with a [hc, n] gain; `value` is [B, T, hc, n].
    GgufValue grouped_norm(const GgufValue& value, const std::string& weight) {
        const auto gain = g.node("GGML_OP_RESHAPE",
                                 {weights->require(weight)},
                                 6,
                                 {{"reshape_target", std::vector<int64_t>{hc, n_embd}}, {"special_zero", false}});
        return g.node("GGML_OP_MUL", {g.node("GGML_OP_RMS_NORM", {value}, 0, {{"eps", eps}}), gain});
    }

    // Hyper-connection read: a data-dependent sigmoid gate on the normalized streams, averaged
    // into one input, and the per-stream write logits of the block output.
    Mix mix(const GgufValue& streams, const std::string& prefix) {
        const auto normalized = flat(grouped_norm(streams, prefix + "norm.weight"));
        const float inv = 1.0f / static_cast<float>(hc);
        GgufValue low, inject;
        if (weights->has(prefix + "down_inject.weight")) {
            const auto joint =
                scale(g.node("GGML_OP_MUL_MAT", {weights->require(prefix + "down_inject.weight"), normalized}), inv);
            OPENVINO_ASSERT(joint.ne(0) == rank + hc, "[GGUF] unexpected rows of ", prefix, "down_inject.weight");
            low = slice(joint, 3, 0, rank);
            inject = slice(joint, 3, rank, hc);
        } else {
            low = scale(g.node("GGML_OP_MUL_MAT", {weights->require(prefix + "down.weight"), normalized}), inv);
            if (weights->has(prefix + "inject.weight"))
                inject =
                    scale(g.node("GGML_OP_MUL_MAT", {weights->require(prefix + "inject.weight"), normalized}), inv);
        }
        const auto gate =
            g.node("GGML_OP_MUL_MAT", {weights->require(prefix + "up.weight"), g.node("GGML_UNARY_OP_SILU", {low})});
        const auto gated =
            reshape(g.node("GGML_OP_MUL", {normalized, g.node("GGML_UNARY_OP_SIGMOID", {gate})}), {0, -1, hc, n_embd});
        auto sum = slice(gated, 2, 0, 1);
        for (int64_t s = 1; s < hc; ++s)
            sum = g.node("GGML_OP_ADD", {sum, slice(gated, 2, s, 1)});
        return {reshape(scale(sum, inv), {0, 1, -1, n_embd}), inject};
    }

    // Hyper-connection write: streams + block_output * 2 * sigmoid(inject), per stream.
    GgufValue combine(const GgufValue& streams, const GgufValue& output, const GgufValue& inject) {
        OPENVINO_ASSERT(inject, "[GGUF] qwen4exp blocks require hyper-connection inject weights");
        const auto weight = reshape(scale(g.node("GGML_UNARY_OP_SIGMOID", {inject}), 2.0f), {0, -1, hc, 1});
        const auto written = g.node("GGML_OP_MUL", {reshape(output, {0, -1, 1, n_embd}), weight});
        return g.node("GGML_OP_ADD", {streams, written});
    }

    void read_ngram(const GgufMetadata& m, const std::string& a, int layers) {
        const auto ple_layers = m.get_int_array(a + "ple.layers");
        if (ple_layers.empty())
            return;
        OPENVINO_ASSERT(ple_layers.size() == 1 && ple_layers[0] >= 0 && ple_layers[0] < layers,
                        "[GGUF] qwen4exp supports exactly one n-gram embedding layer");
        const auto value = [&](const std::string& key) {
            const auto v = m.get_int(a + key);
            OPENVINO_ASSERT(v && *v >= 0, "[GGUF] qwen4exp n-gram embeddings require ", a + key);
            return *v;
        };
        ngram.layer = static_cast<int>(ple_layers[0]);
        ngram.ngram_size = value("ple.ngram_size");
        ngram.heads_per_ngram = value("ple.heads_per_ngram");
        ngram.conv_kernel = value("ple.conv_kernel");
        ngram.eos = value("ple.eos_token_id");
        ngram.head_dim = value("embedding_length_per_layer_input");
        ngram.multipliers = m.get_int_array(a + "ple.layer_multipliers");
        ngram.offsets = m.get_int_array(a + "ple.head_offsets");
        ngram.vocab_sizes = m.get_int_array(a + "ple.head_vocab_sizes");
        const auto heads = static_cast<size_t>((ngram.ngram_size - 1) * ngram.heads_per_ngram);
        OPENVINO_ASSERT(ngram.ngram_size >= 2 && heads > 0 && ngram.conv_kernel > 0 && ngram.head_dim > 0 &&
                            ngram.multipliers.size() >= static_cast<size_t>(ngram.ngram_size) &&
                            ngram.offsets.size() >= heads && ngram.vocab_sizes.size() >= heads,
                        "[GGUF] invalid qwen4exp n-gram embedding metadata");
    }

    // Rows of the hashed n-gram table for every token, gathered and flattened to [B, 1, T, heads * dim].
    // The previous ngram_size - 1 tokens are a state, stored relative to EOS so that a reset
    // (zero) state starts a sequence.
    GgufValue build_ngram_embeddings() {
        const int64_t context = ngram.ngram_size - 1;
        const int64_t heads = context * ngram.heads_per_ngram;
        const auto history = g.add_input("ngram_history", i32, {1, context});
        auto hashed = g.node_outputs("GGUF_OP_NGRAM_HASH",
                                     {g.add_input("inp_tokens", i32, ov::PartialShape({1, 1, 1, -1})), history},
                                     2,
                                     0,
                                     {{"ngram_size", ngram.ngram_size},
                                      {"heads_per_ngram", ngram.heads_per_ngram},
                                      {"multipliers", ngram.multipliers},
                                      {"head_vocab_sizes", ngram.vocab_sizes},
                                      {"head_offsets", ngram.offsets},
                                      {"eos_token_id", ngram.eos},
                                      {"history_offset", ngram.eos}});
        g.add_recurrent_state(history, hashed[1]);
        const auto table = weights->require("per_layer_token_embd.weight");
        OPENVINO_ASSERT(table.ne(0) == ngram.head_dim, "[GGUF] n-gram table rows do not match its head dimension");
        return reshape(g.node("GGML_OP_GET_ROWS", {table, hashed[0]}), {0, 1, -1, heads * ngram.head_dim});
    }

    // Gated n-gram embedding added to the streams: a per-stream sigmoid gate from a signed square
    // root of key/query similarity scales the value, which also feeds a causal depthwise
    // convolution dilated by the n-gram size.
    GgufValue build_ple(const std::string& prefix, const GgufValue& streams, const GgufValue& embeddings, int layer) {
        const auto key = reshape(g.node("GGML_OP_MUL_MAT", {weights->require(prefix + "ple_key.weight"), embeddings}),
                                 {0, -1, hc, n_embd});
        const auto value = g.node("GGML_OP_MUL_MAT", {weights->require(prefix + "ple_value.weight"), embeddings});
        const auto similarity = scale(g.node("GGML_OP_SUM_ROWS",
                                             {g.node("GGML_OP_MUL",
                                                     {grouped_norm(key, prefix + "ple_norm_key.weight"),
                                                      grouped_norm(streams, prefix + "ple_norm_query.weight")})}),
                                      1.0f / std::sqrt(static_cast<float>(n_embd)));
        const auto magnitude =
            g.node("GGML_OP_SQRT",
                   {g.node("GGML_OP_CLAMP",
                           {g.node("GGML_UNARY_OP_ABS", {similarity})},
                           0,
                           {{"clamp_min", 1e-6f}, {"clamp_max", std::numeric_limits<float>::infinity()}})});
        const auto gate = g.node("GGML_UNARY_OP_SIGMOID",
                                 {g.node("GGML_OP_MUL", {g.node("GGML_UNARY_OP_SGN", {similarity}), magnitude})});
        const auto gated = g.node("GGML_OP_MUL", {reshape(value, {0, -1, 1, n_embd}), gate});

        // Depthwise conv over [history | tokens]: out[t] = sum_k w[k] * x[t - (K - 1 - k) * dilation],
        // a dense kernel of (K - 1) * dilation + 1 taps that is zero between the dilated ones.
        const int64_t channels = hc * n_embd;
        const int64_t dilation = ngram.ngram_size;
        const int64_t history = (ngram.conv_kernel - 1) * dilation;
        const auto kernel = weights->require(prefix + "ple_conv1d.weight");
        OPENVINO_ASSERT(kernel.ne(0) == ngram.conv_kernel && kernel.ne(1) == channels,
                        "[GGUF] unexpected n-gram convolution kernel in layer ",
                        layer);
        ov::Tensor spread(f32, {1, 1, static_cast<size_t>(history + 1), static_cast<size_t>(ngram.conv_kernel)});
        std::fill_n(spread.data<float>(), spread.get_size(), 0.0f);
        for (int64_t k = 0; k < ngram.conv_kernel; ++k)
            spread.data<float>()[k * dilation * ngram.conv_kernel + k] = 1.0f;
        const auto dense_kernel =
            g.node("GGML_OP_MUL_MAT", {g.add_constant(prefix + "ple_conv_spread", spread), kernel});

        const auto normalized =
            g.node("GGML_OP_TRANSPOSE", {flat(grouped_norm(gated, prefix + "ple_norm_conv.weight"))});
        const auto state = g.add_input("ple_conv_state_l" + std::to_string(layer), f32, {1, 1, channels, history});
        const auto window = g.node("GGML_OP_CONCAT", {state, normalized}, 0, {{"concat_axis", 0}});
        g.add_recurrent_state(state, slice(window, 3, -history, history));
        const auto conv =
            g.node("GGML_OP_SSM_CONV", {window, dense_kernel}, 0, {{"batch_major", true}, {"fuse_silu", true}});
        const auto added = g.node("GGML_OP_ADD", {gated, reshape(conv, {0, -1, hc, n_embd})});
        return g.node("GGML_OP_ADD", {streams, added});
    }

    GgufGraphContext g;
    std::unique_ptr<GgufTensors> weights;
    NgramConfig ngram;
    int64_t hc = 0, rank = 0, n_embd = 0;
    float eps = 0.0f;
};

}  // namespace

std::shared_ptr<ModelBuilder> make_qwen4exp_builder(const BuildContext& context) {
    return std::make_shared<Qwen4ExpBuilder>(context);
}

}  // namespace ov::frontend::gguf
