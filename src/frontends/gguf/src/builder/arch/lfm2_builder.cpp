// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lfm2_builder.hpp"

#include <cmath>
#include <limits>

#include "builder/decoder_metadata.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov::frontend::gguf {
namespace {
class Lfm2Builder : public ModelBuilder {
public:
    explicit Lfm2Builder(const BuildContext& context) : g(context) {}

    std::shared_ptr<GgufGraph> build() override {
        const auto kernel = g.metadata().get_int(g.arch() + ".shortconv.l_cache");
        OPENVINO_ASSERT(kernel && *kernel > 1 && *kernel <= std::numeric_limits<int>::max(),
                        "[GGUF] LFM2 requires shortconv.l_cache > 1");
        OPENVINO_ASSERT(g.metadata().get_bool(g.arch() + ".attention.causal").value_or(true),
                        "[GGUF] noncausal LFM2 embedding models require a centered convolution builder");
        DecoderOptions options;
        options.normalize_expert_weights = true;
        const auto parameters = resolve_decoder_parameters(g.metadata(), RopeMode::Neox, options);
        const auto dims = g.configure_decoder(parameters);
        OPENVINO_ASSERT(dims.layers > 0 && dims.embedding > 0,
                        "[GGUF] LFM2 requires positive layer and embedding counts");
        g.build_inp_pos();
        g.build_attn_inp_kv();
        auto weights = g.tensors();
        auto cur = g.build_inp_embd(weights.require("token_embd.weight"));
        const bool detokenizer = weights.has("dense_2.weight");
        if (detokenizer)
            OPENVINO_ASSERT(weights.require("dense_2.weight").ne(0) == dims.embedding &&
                                weights.require("dense_2.weight").ne(1) == 1282,
                            "[GGUF] LFM2 audio detokenizer requires a 1282-channel spectral head");
        const auto out_ids = detokenizer ? GgufValue{} : g.build_inp_out_ids();
        for (int layer = 0; layer < dims.layers; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto w = [&](const std::string& name) {
                return weights.require(prefix + name);
            };
            auto residual = cur;
            cur = g.build_norm(cur, w("attn_norm.weight"), dims.norm_epsilon);
            if (weights.has(prefix + "shortconv.in_proj.weight")) {
                const auto projected = g.node("GGML_OP_MUL_MAT", {w("shortconv.in_proj.weight"), cur});
                const auto b = slice(projected, 0, dims.embedding);
                const auto c = slice(projected, dims.embedding, dims.embedding);
                const auto x = slice(projected, 2 * int64_t{dims.embedding}, dims.embedding);
                const auto bx = g.node("GGML_OP_TRANSPOSE", {g.node("GGML_OP_MUL", {b, x})});
                const auto state = g.add_input("conv_state_l" + std::to_string(layer),
                                               ov::element::f32,
                                               {1, 1, dims.embedding, *kernel - 1});
                const auto window = g.node("GGML_OP_CONCAT", {state, bx}, 0, {{"concat_axis", 0}});
                g.add_recurrent_state(state, slice(window, -(*kernel - 1), *kernel - 1));
                const auto convolution =
                    g.node("GGML_OP_SSM_CONV", {window, w("shortconv.conv.weight")}, 0, {{"batch_major", true}});
                cur = g.node("GGML_OP_MUL_MAT",
                             {w("shortconv.out_proj.weight"), g.node("GGML_OP_MUL", {c, convolution})});
            } else {
                cur = g.decoder_attention(layer, cur);
            }
            if (layer == dims.layers - 1 && out_ids) {
                cur = g.node("GGML_OP_GET_ROWS", {cur, out_ids});
                residual = g.node("GGML_OP_GET_ROWS", {residual, out_ids});
            }
            cur = g.node("GGML_OP_ADD", {cur, residual});
            residual = cur;
            cur = g.build_norm(cur, w("ffn_norm.weight"), dims.norm_epsilon);
            cur = g.node("GGML_OP_ADD", {residual, g.decoder_ffn(layer, cur)});
        }
        cur = g.build_norm(cur, weights.require("token_embd_norm.weight"), dims.norm_epsilon);
        if (detokenizer) {
            auto output = g.node("GGML_OP_MUL_MAT", {weights.require("dense_2.weight"), cur});
            if (auto bias = weights("dense_2.bias"))
                output = g.node("GGML_OP_ADD", {output, bias});
            g.set_primary_output(output);
        } else {
            const auto output =
                weights.has("output.weight") ? weights.require("output.weight") : weights.require("token_embd.weight");
            g.set_primary_output(g.node("GGML_OP_MUL_MAT", {output, cur}));
        }
        return g.finish();
    }

private:
    GgufValue slice(const GgufValue& value, int64_t start, int64_t length) {
        return g.node("GGML_OP_VIEW", {value}, 3, {{"view_slice", std::vector<int64_t>{3, start, length}}});
    }

    GgufGraphContext g;
};
class Lfm2VocoderBuilder : public ModelBuilder {
public:
    explicit Lfm2VocoderBuilder(const BuildContext& context) : g(context) {}

    std::shared_ptr<GgufGraph> build() override {
        auto weights = g.tensors();
        const auto width = g.metadata().get_int("depthformer_n_embd").value_or(0);
        const auto layers = g.metadata().get_int("depthformer_n_layer").value_or(0);
        OPENVINO_ASSERT(width > 0 && layers > 0, "[GGUF] invalid LFM2 depthformer dimensions");
        const auto projection = weights.require("depth_linear.weight");
        constexpr int64_t books = 8;
        const auto vocabulary = weights.require("depth_embeddings.0.to_logits.weight").ne(1);
        const auto head = weights.require("depthformer.layers.0.operator.attention.q_layernorm.weight").ne(0);
        const auto qkv_width = weights.require("depthformer.layers.0.operator.qkv_proj.weight").ne(1);
        const auto kv_width = (qkv_width - width) / 2;
        OPENVINO_ASSERT(projection.ne(1) == books * width && vocabulary == 2049 && head > 0 && width % head == 0 &&
                            kv_width > 0 && kv_width % head == 0 && width % kv_width == 0,
                        "[GGUF] invalid LFM2 audio depthformer tensor dimensions");
        RopeConfig rope;
        rope.n_dims = int(head);
        rope.n_ctx_orig = 128000;
        rope.freq_base = 1000000.f;
        rope.freq_scale = 1.f;
        rope.attn_factor = 1.f;
        rope.per_op = true;
        g.configure_rope(rope);
        auto hidden = g.add_input("audio.hidden_states", ov::element::f32, {1, 1, 1, projection.ne(0)});
        auto previous = g.add_input("audio.depth_codes", ov::element::i32, {1, 1, 1, books});
        auto x = reshape(linear(hidden, "depth_linear"), {1, 1, books, width});
        ov::Tensor zeros(ov::element::f32, {1, 1, 1, size_t(width)});
        std::fill_n(zeros.data<float>(), zeros.get_size(), 0.f);
        auto shifted = g.add_constant("audio.depth_zero", zeros);
        for (int64_t book = 1; book < books; ++book) {
            auto embedding =
                g.node("GGML_OP_GET_ROWS",
                       {weights.require("depth_embeddings." + std::to_string(book - 1) + ".embedding.weight"),
                        slice(previous, 3, book - 1, 1)});
            shifted = concat(shifted, embedding, 1);
        }
        x = add(x, shifted);
        ov::Tensor positions(ov::element::i32, {1, 1, 1, size_t(books)});
        ov::Tensor mask(ov::element::f32, {1, 1, size_t(books), size_t(books)});
        for (int64_t i = 0; i < books; ++i) {
            positions.data<int32_t>()[i] = int32_t(i);
            for (int64_t j = 0; j < books; ++j)
                mask.data<float>()[i * books + j] = j <= i ? 0.f : -std::numeric_limits<float>::infinity();
        }
        auto pos = g.add_constant("audio.depth_positions", positions);
        auto causal = g.add_constant("audio.depth_mask", mask);
        for (int64_t i = 0; i < layers; ++i) {
            const auto p = "depthformer.layers." + std::to_string(i) + ".";
            auto normalized = g.build_norm(x, weights.require(p + "operator_norm.weight"), 1e-5f);
            auto qkv = linear(normalized, p + "operator.qkv_proj");
            const auto heads = [&](int64_t begin, int64_t channels, const std::string& norm) {
                auto value = reshape(slice(qkv, 3, begin, channels), {1, books, channels / head, head});
                if (!norm.empty())
                    value = g.build_norm(value, weights.require(p + norm), 1e-5f);
                return value;
            };
            auto q = heads(0, width, "operator.attention.q_layernorm.weight");
            auto k = heads(width, kv_width, "operator.attention.k_layernorm.weight");
            auto v = heads(width + kv_width, kv_width, "");
            q = g.node("GGML_OP_ROPE", {q, pos}, 0, {{"rope_config", rope}});
            k = g.node("GGML_OP_ROPE", {k, pos}, 0, {{"rope_config", rope}});
            auto attention = g.node("GGML_OP_FLASH_ATTN_EXT",
                                    {q, k, v, causal},
                                    100,
                                    {{"f32_attention", true}, {"scale", 1.f / std::sqrt(float(head))}});
            x = add(x, linear(reshape(attention, {1, 1, books, width}), p + "operator.out_proj"));
            normalized = g.build_norm(x, weights.require(p + "ffn_norm.weight"), 1e-5f);
            auto gated = g.node("GGML_OP_MUL",
                                {g.node("GGML_UNARY_OP_SILU", {linear(normalized, p + "feed_forward.w1")}),
                                 linear(normalized, p + "feed_forward.w3")});
            x = add(x, linear(gated, p + "feed_forward.w2"));
        }
        GgufValue logits;
        for (int64_t book = 0; book < books; ++book) {
            const auto p = "depth_embeddings." + std::to_string(book);
            auto normalized = g.build_norm(slice(x, 2, book, 1), weights.require(p + ".embedding_norm.weight"), 1e-5f);
            auto current = linear(normalized, p + ".to_logits");
            logits = logits ? concat(logits, current, 1) : current;
        }
        g.set_output(logits, "audio.depth_logits");
        auto codes = g.add_input("audio.codes", ov::element::i32, {1, 1, -1, books});
        GgufValue feedback, codec;
        for (int64_t book = 0; book < books; ++book) {
            auto index = reshape(slice(codes, 3, book, 1), {1, 1, 1, -1});
            const auto lookup = [&](const std::string& name, int64_t stride) {
                ov::Tensor offset(ov::element::i32, {1, 1, 1, 1});
                offset.data<int32_t>()[0] = int32_t(book * stride);
                auto ids = add(index, g.add_constant(name + ".offset." + std::to_string(book), offset));
                return g.node("GGML_OP_GET_ROWS", {weights.require(name), ids});
            };
            auto audio = lookup("audio_embedding.embedding.weight", vocabulary);
            auto frame = lookup("emb.emb.weight", vocabulary - 1);
            feedback = feedback ? add(feedback, audio) : audio;
            codec = codec ? add(codec, frame) : frame;
        }
        g.set_output(feedback, "audio.feedback");
        g.set_output(g.node("GGML_OP_SCALE", {codec}, 0, {{"scale", 1.f / float(books)}}), "audio.codec_features");
        g.set_output(weights.require("istft.window"), "audio.istft_window");
        return g.finish();
    }

private:
    GgufValue reshape(const GgufValue& value, const std::vector<int64_t>& shape) {
        return g.node("GGML_OP_RESHAPE", {value}, 0, {{"reshape_target", shape}});
    }
    GgufValue slice(const GgufValue& value, int64_t axis, int64_t start, int64_t length) {
        return g.node("GGML_OP_VIEW", {value}, 3, {{"view_slice", std::vector<int64_t>{axis, start, length}}});
    }
    GgufValue add(const GgufValue& a, const GgufValue& b) {
        return g.node("GGML_OP_ADD", {a, b});
    }
    GgufValue concat(const GgufValue& a, const GgufValue& b, int axis) {
        return g.node("GGML_OP_CONCAT", {a, b}, 0, {{"concat_axis", axis}});
    }
    GgufValue linear(const GgufValue& value, const std::string& name) {
        auto result = g.node("GGML_OP_MUL_MAT", {g.tensors().require(name + ".weight"), value});
        if (auto bias = g.tensors()(name + ".bias"))
            result = add(result, bias);
        return result;
    }
    GgufGraphContext g;
};
}  // namespace

std::shared_ptr<ModelBuilder> make_lfm2_builder(const BuildContext& context) {
    return std::make_shared<Lfm2Builder>(context);
}

std::shared_ptr<ModelBuilder> make_lfm2_vocoder_builder(const BuildContext& context) {
    return std::make_shared<Lfm2VocoderBuilder>(context);
}
}  // namespace ov::frontend::gguf
