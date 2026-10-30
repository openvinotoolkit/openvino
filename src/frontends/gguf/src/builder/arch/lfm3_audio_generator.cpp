// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lfm3_audio_generator.hpp"

#include <cmath>

#include "lfm2_builder.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov::frontend::gguf {
namespace {
class AudioGeneration {
public:
    explicit AudioGeneration(GgufGraphContext& context) : g(context) {}

    void build() {
        const auto embedding = g.tensors().require("a.gen.code.embd.weight");
        width = embedding.ne(0);
        vocabulary = embedding.ne(1);
        books = embedding.ne(2);
        OPENVINO_ASSERT(width > 0 && books == 12 && vocabulary == 2017,
                        "[GGUF] LFM3 audio generation requires twelve 2017-entry codebooks");
        depth();
        feedback();
        codec();
        ov::Tensor window(ov::element::f32, {1280});
        for (size_t i = 0; i < window.get_size(); ++i)
            window.data<float>()[i] = .5f * (1.f - std::cos(2.f * std::acos(-1.f) * float(i) / 1280.f));
        g.set_output(g.add_constant("audio.codec.istft_window", window), "audio.istft_window");
    }

private:
    GgufValue reshape(const GgufValue& x, const std::vector<int64_t>& shape) {
        return g.node("GGML_OP_RESHAPE", {x}, 0, {{"reshape_target", shape}});
    }
    GgufValue slice(const GgufValue& x, int axis, int64_t start, int64_t length) {
        return g.node("GGML_OP_VIEW", {x}, 3, {{"view_slice", std::vector<int64_t>{axis, start, length}}});
    }
    GgufValue concat(const GgufValue& x, const GgufValue& y, int axis) {
        return g.node("GGML_OP_CONCAT", {x, y}, 0, {{"concat_axis", axis}});
    }
    GgufValue add(const GgufValue& x, const GgufValue& y) {
        return g.node("GGML_OP_ADD", {x, y});
    }
    GgufValue mul(const GgufValue& x, const GgufValue& y) {
        return g.node("GGML_OP_MUL", {x, y});
    }
    GgufValue linear(const GgufValue& x, const std::string& name) {
        auto y = g.node("GGML_OP_MUL_MAT", {g.tensors().require(name + ".weight"), x});
        if (auto b = g.tensors()(name + ".bias"))
            y = add(y, b);
        return y;
    }
    GgufValue norm(const GgufValue& x, const std::string& name, float epsilon) {
        return g.build_norm(x, g.tensors().require(name + ".weight"), epsilon);
    }
    GgufValue book_ids(const GgufValue& codes, int64_t book, const std::string& prefix, bool offset) {
        auto index = reshape(slice(codes, 3, book, 1), {1, 1, 1, -1});
        if (offset) {
            ov::Tensor value(ov::element::i32, {1, 1, 1, 1});
            value.data<int32_t>()[0] = int32_t(book * vocabulary);
            index = add(index, g.add_constant(prefix + ".offset." + std::to_string(book), value));
        }
        return index;
    }
    GgufValue attention(const GgufValue& x,
                        const std::string& prefix,
                        float epsilon,
                        const GgufValue& positions,
                        const GgufValue& mask = {},
                        int layer = -1) {
        const auto head = g.tensors().require(prefix + "attn_q_norm.weight").ne(0);
        const auto qwidth = g.tensors().require(prefix + "attn_q.weight").ne(1);
        const auto kvwidth = g.tensors().require(prefix + "attn_k.weight").ne(1);
        OPENVINO_ASSERT(head > 0 && head % 2 == 0 && qwidth % head == 0 && kvwidth > 0 && kvwidth % head == 0 &&
                            qwidth % kvwidth == 0 && g.tensors().require(prefix + "attn_v.weight").ne(1) == kvwidth,
                        "[GGUF] invalid LFM3 audio attention dimensions at ",
                        prefix);
        const bool streaming = layer >= 0;
        const auto heads = [&](const GgufValue& value, int64_t count, int64_t size) {
            return streaming ? reshape(value, {1, -1, count, size}) : reshape(value, {-1, books, count, size});
        };
        auto q = norm(heads(linear(x, prefix + "attn_q"), qwidth / head, head), prefix + "attn_q_norm", epsilon);
        auto k = norm(heads(linear(x, prefix + "attn_k"), kvwidth / head, head), prefix + "attn_k_norm", epsilon);
        auto v = heads(linear(x, prefix + "attn_v"), kvwidth / head, head);
        RopeConfig rope;
        rope.n_dims = int(head);
        rope.freq_base = 1000000.f;
        rope.freq_scale = 1.f;
        rope.attn_factor = 1.f;
        rope.per_op = true;
        q = g.node("GGML_OP_ROPE", {q, positions}, 1 << 16, {{"rope_config", rope}});
        k = g.node("GGML_OP_ROPE", {k, positions}, 1 << 16, {{"rope_config", rope}});
        if (streaming) {
            const auto state = [&](const std::string& kind, const GgufValue& current) {
                auto input = g.add_input("audio.codec.state." + kind + "." + std::to_string(layer),
                                         ov::element::f32,
                                         {1, 1, 29, kvwidth});
                auto all = concat(input, reshape(current, {1, 1, -1, kvwidth}), 1);
                g.add_recurrent_state(input, slice(all, 2, -29, 29));
                return heads(all, kvwidth / head, head);
            };
            k = state("k", k);
            v = state("v", v);
        }
        std::vector<GgufValue> inputs{q, k, v};
        if (mask)
            inputs.push_back(mask);
        auto output = g.node("GGML_OP_FLASH_ATTN_EXT",
                             inputs,
                             100,
                             {{"f32_attention", true}, {"scale", 1.f / std::sqrt(float(head))}});
        if (auto gate_weight = g.tensors()(prefix + "attn_gate.weight")) {
            OPENVINO_ASSERT(gate_weight.ne(1) == qwidth / head,
                            "[GGUF] LFM3 audio requires one sigmoid gate per query head");
            auto gate = g.node("GGML_UNARY_OP_SIGMOID", {linear(x, prefix + "attn_gate")});
            output = mul(output, heads(gate, qwidth / head, 1));
        }
        output = streaming ? reshape(output, {1, 1, -1, qwidth}) : reshape(output, {-1, 1, books, qwidth});
        return linear(output, prefix + "attn_out");
    }
    GgufValue convolution(const GgufValue& x, const std::string& prefix, int layer) {
        const auto channels = x.ne(0);
        const auto kernel = g.tensors().require(prefix + "conv_dw.weight").ne(0);
        OPENVINO_ASSERT(kernel > 1 && g.tensors().require(prefix + "conv_pw1.weight").ne(1) == 3 * channels,
                        "[GGUF] invalid LFM3 codec short convolution");
        const auto gated = build_lfm_short_conv(g,
                                                linear(x, prefix + "conv_pw1"),
                                                g.tensors().require(prefix + "conv_dw.weight"),
                                                channels,
                                                "audio.codec.state.conv." + std::to_string(layer));
        return linear(gated, prefix + "conv_pw2");
    }
    GgufValue block(const GgufValue& x,
                    const std::string& prefix,
                    float epsilon,
                    const GgufValue& positions,
                    const GgufValue& mask = {},
                    int layer = -1) {
        const bool conv = g.tensors().has(prefix + "conv_dw.weight");
        OPENVINO_ASSERT(conv != g.tensors().has(prefix + "attn_q.weight") && (!conv || layer >= 0),
                        "[GGUF] LFM3 audio blocks require exactly one supported operator");
        OPENVINO_ASSERT(
            g.tensors().has(prefix + "attn_post_norm.weight") == g.tensors().has(prefix + "ffn_post_norm.weight"),
            "[GGUF] LFM3 audio sandwich norms must appear as a pair");
        auto y = norm(x, prefix + "ln1", epsilon);
        y = conv ? convolution(y, prefix, layer) : attention(y, prefix, epsilon, positions, mask, layer);
        if (g.tensors().has(prefix + "attn_post_norm.weight"))
            y = norm(y, prefix + "attn_post_norm", epsilon);
        auto residual = add(x, y);
        y = norm(residual, prefix + "ln2", epsilon);
        y = mul(g.node("GGML_UNARY_OP_SILU", {linear(y, prefix + "ffn_gate")}), linear(y, prefix + "ffn_up"));
        y = linear(y, prefix + "ffn_down");
        if (g.tensors().has(prefix + "ffn_post_norm.weight"))
            y = norm(y, prefix + "ffn_post_norm", epsilon);
        return add(residual, y);
    }
    void depth() {
        const auto projection = g.tensors().require("a.gen.code.proj_in.weight");
        OPENVINO_ASSERT(projection.ne(1) == books * width, "[GGUF] invalid LFM3 depth projection");
        const auto hidden = g.add_input("audio.depth.hidden_states", ov::element::f32, {1, 1, -1, projection.ne(0)});
        const auto codes = g.add_input("audio.depth.codes", ov::element::i32, {1, 1, -1, books});
        const auto mask = g.add_input("audio.depth.mask", ov::element::f32, {1, 1, -1, books});
        const auto tables = reshape(g.tensors().require("a.gen.code.embd.weight"), {1, books, vocabulary, width});
        GgufValue embedded;
        for (int64_t book = 0; book < books; ++book) {
            auto token =
                g.node("GGML_OP_GET_ROWS", {slice(tables, 1, book, 1), book_ids(codes, book, "audio.depth", false)});
            auto revealed = slice(mask, 3, book, 1);
            token = add(mul(token, revealed),
                        mul(g.tensors().require("a.gen.code.mask_embd"),
                            g.node("GGML_OP_SCALE", {revealed}, 0, {{"scale", -1.f}, {"bias", 1.f}})));
            token = reshape(token, {-1, 1, 1, width});
            embedded = embedded ? concat(embedded, token, 1) : token;
        }
        auto current = add(reshape(linear(hidden, "a.gen.code.proj_in"), {-1, 1, books, width}), embedded);
        ov::Tensor positions(ov::element::i32, {1, 1, 1, size_t(books)});
        for (int64_t i = 0; i < books; ++i)
            positions.data<int32_t>()[i] = int32_t(i);
        const auto pos = g.add_constant("audio.depth.positions", positions);
        const auto layers = g.metadata().get_int("clip.gen.audio.block_count").value_or(0);
        const auto epsilon =
            float(g.metadata().get_float("clip.gen.audio.attention.layer_norm_epsilon").value_or(1e-5));
        OPENVINO_ASSERT(layers == 3, "[GGUF] LFM3 depthformer requires three bidirectional attention layers");
        for (int layer = 0; layer < layers; ++layer)
            current = block(current, "a.gen.code.blk." + std::to_string(layer) + ".", epsilon, pos);
        current = norm(current, "a.gen.code.output_norm", epsilon);
        const auto head_norms = reshape(g.tensors().require("a.gen.code.head_norm.weight"), {1, 1, books, width});
        GgufValue logits;
        for (int64_t book = 0; book < books; ++book) {
            const auto value = g.build_norm(slice(current, 2, book, 1), slice(head_norms, 2, book, 1), epsilon);
            auto output = g.node("GGML_OP_MUL_MAT", {slice(tables, 1, book, 1), value});
            logits = logits ? concat(logits, output, 1) : output;
        }
        g.set_output(reshape(logits, {1, -1, books, vocabulary}), "audio.depth_logits");
    }
    void feedback() {
        const auto codes = g.add_input("audio.feedback.codes", ov::element::i32, {1, 1, -1, books});
        const auto table = g.tensors().require("a.gen.code.out_embd.weight");
        OPENVINO_ASSERT(table.ne(1) == books * vocabulary, "[GGUF] invalid LFM3 feedback embedding table");
        GgufValue sum;
        for (int64_t book = 0; book < books; ++book) {
            auto value = g.node("GGML_OP_GET_ROWS", {table, book_ids(codes, book, "audio.feedback", true)});
            sum = sum ? add(sum, value) : value;
        }
        g.set_output(sum, "audio.feedback");
    }
    void codec() {
        const auto table = g.tensors().require("a.gen.wav.quant.first.codebook.weight");
        OPENVINO_ASSERT(table.ne(0) == 4 && table.ne(1) == vocabulary - 1,
                        "[GGUF] LFM3 codec requires the [8,7,6,6] finite scalar codebook");
        const auto codes = g.add_input("audio.codec.codes", ov::element::i32, {1, 1, -1, books});
        GgufValue quantized;
        for (int64_t book = 0; book < books; ++book) {
            auto value = g.node("GGML_OP_GET_ROWS", {table, book_ids(codes, book, "audio.codec", false)});
            quantized = quantized ? concat(quantized, value, 0) : value;
        }
        auto current = linear(linear(quantized, "a.gen.wav.quant.first.out_proj"), "a.gen.wav.tfm.in_proj");
        const auto channels = current.ne(0);
        current = g.node("GGML_OP_REPEAT",
                         {reshape(current, {1, -1, 1, channels})},
                         0,
                         {{"repeats", std::vector<int64_t>{1, 1, 6, 1}}});
        current = reshape(current, {1, 1, -1, channels});
        const auto positions = g.add_input("audio.codec.positions", ov::element::i32, {1, 1, 1, -1});
        const auto mask = g.add_input("audio.codec.attention_mask", ov::element::f32, {1, 1, -1, -1});
        const auto position = g.add_input("audio.codec.state.pos", ov::element::f32, {1, 1, 1, 1});
        auto count = g.node("GGML_OP_TRANSPOSE", {slice(current, 3, 0, 1)});
        count = g.node("GGML_OP_SUM_ROWS", {g.node("GGML_OP_FILL", {count}, 0, {{"fill_value", 1.f}})});
        g.add_recurrent_state(position, add(position, count));
        int layer = 0;
        for (; g.tensors().has("a.gen.wav.tfm.blk." + std::to_string(layer) + ".ln1.weight"); ++layer)
            current = block(current, "a.gen.wav.tfm.blk." + std::to_string(layer) + ".", 1e-5f, positions, mask, layer);
        OPENVINO_ASSERT(layer == 8, "[GGUF] LFM3 codec requires eight streaming decoder layers");
        current = linear(norm(current, "a.gen.wav.tfm.output_norm", 1e-5f), "a.gen.wav.tfm.out_proj");
        OPENVINO_ASSERT(current.ne(0) == 1282, "[GGUF] LFM3 codec requires a 1282-channel polar spectral output");
        g.set_output(current, "audio.spectrogram");
    }
    GgufGraphContext& g;
    int64_t width = 0, vocabulary = 0, books = 0;
};
}  // namespace

void build_lfm3_audio_generation(GgufGraphContext& g) {
    AudioGeneration(g).build();
}
}  // namespace ov::frontend::gguf
