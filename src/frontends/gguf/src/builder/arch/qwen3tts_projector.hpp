// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "quant/gguf.hpp"

namespace ov::frontend::gguf {

class Qwen3TTSProjector {
public:
    explicit Qwen3TTSProjector(GgufGraphContext& graph) : g(graph) {}

    void build() {
        code_predictor();
        waveform();
    }

private:
    GgufGraphContext& g;
    size_t constant_id = 0;

    GgufValue constant(const std::vector<float>& values, const ov::Shape& shape) {
        ov::Tensor tensor(ov::element::f32, shape);
        std::copy(values.begin(), values.end(), tensor.data<float>());
        return g.add_constant("tts.constant." + std::to_string(constant_id++), tensor);
    }
    GgufValue scalar(float value) {
        return constant({value}, {1, 1, 1, 1});
    }
    GgufValue reshape(const GgufValue& x, std::vector<int64_t> shape) {
        return g.node("GGML_OP_RESHAPE", {x}, 0, {{"reshape_target", std::move(shape)}});
    }
    GgufValue transpose(const GgufValue& x, std::vector<int64_t> order) {
        return g.node("GGML_OP_TRANSPOSE", {x}, 0, {{"perm", std::move(order)}});
    }
    GgufValue slice(const GgufValue& x, int64_t axis, int64_t begin, int64_t count) {
        return g.node("GGML_OP_VIEW", {x}, 3, {{"view_slice", std::vector<int64_t>{axis, begin, count}}});
    }
    GgufValue cast(const GgufValue& x, ov::element::Type type) {
        return g.node("GGML_OP_CPY", {x}, 0, {{"dst_type", type}});
    }
    GgufValue add(const GgufValue& a, const GgufValue& b) {
        return g.node("GGML_OP_ADD", {a, b});
    }
    GgufValue mul(const GgufValue& a, const GgufValue& b) {
        return g.node("GGML_OP_MUL", {a, b});
    }
    GgufValue scale(const GgufValue& x, float factor, float bias = 0) {
        return g.node("GGML_OP_SCALE", {x}, 0, {{"scale", factor}, {"bias", bias}});
    }
    GgufValue concat(const GgufValue& a, const GgufValue& b, int64_t axis) {
        return g.node("GGML_OP_CONCAT", {a, b}, 0, {{"concat_axis", int(a.shape().rank().get_length() - 1 - axis)}});
    }
    GgufValue linear(const GgufValue& x, const std::string& base) {
        auto y = g.node("GGML_OP_MUL_MAT", {g.tensors().require(base + ".weight"), x});
        if (auto bias = g.tensors()(base + ".bias"))
            y = add(y, bias);
        return y;
    }
    GgufValue norm(const GgufValue& x, const std::string& base, float eps) {
        return g.build_norm(x, g.tensors().require(base + ".weight"), eps);
    }
    GgufValue ffn(const GgufValue& x, const std::string& base) {
        return linear(mul(g.node("GGML_UNARY_OP_SILU", {linear(x, base + "ffn_gate")}), linear(x, base + "ffn_up")),
                      base + "ffn_down");
    }
    GgufValue rope(const GgufValue& x, const GgufValue& pos, float theta, int64_t width) {
        RopeConfig r;
        r.n_dims = int(width);
        r.freq_base = theta;
        r.freq_scale = r.attn_factor = 1.f;
        r.per_op = true;
        return g.node("GGML_OP_ROPE", {x, pos}, 1 << 16, {{"rope_config", r}});
    }
    GgufValue rows(const GgufValue& table, const GgufValue& ids) {
        return g.node("GGML_OP_GET_ROWS", {table, reshape(ids, {1, 1, 1, -1})});
    }
    GgufValue table(const GgufValue& stacked, int64_t group) {
        return reshape(slice(stacked, stacked.shape().rank().get_length() - 3, group, 1),
                       {1, 1, stacked.ne(1), stacked.ne(0)});
    }
    GgufValue sample(const GgufValue& logits, const GgufValue& random) {
        const auto vocab = logits.ne(0);
        const auto top_k = g.metadata().get_int("general.sampling.top_k").value_or(50);
        const auto top_p = g.metadata().get_float("general.sampling.top_p").value_or(1.0);
        OPENVINO_ASSERT(top_p > 0 && top_p <= 1 && top_k >= 0, "[GGUF] invalid Qwen3TTS sampling defaults");
        const auto count = top_k > 0 ? std::min(top_k, vocab) : vocab;
        const bool filter_k = top_k > 0 && top_k < vocab;
        std::vector<float> vocabulary(static_cast<size_t>(vocab));
        for (int64_t i = 0; i < vocab; ++i)
            vocabulary[size_t(i)] = float(i);
        auto indices = filter_k || top_p < 1 ? g.node("GGML_OP_TOP_K", {logits}, 0, {{"k", count}})
                                             : cast(constant(vocabulary, {1, 1, 1, size_t(vocab)}), ov::element::i32);
        // Match the pinned ggml CPU candidate order used by on-device sampling.
        if (top_p == 1 && count > 1 && filter_k) {
            auto ordered = indices;
            auto first = slice(indices, 3, 0, 1);
            auto second = slice(indices, 3, 1, 1);
            indices = concat(second, first, 3);
            if (count > 2)
                indices = concat(indices, slice(ordered, 3, 2, count - 2), 3);
        }
        auto sorted = reshape(rows(reshape(logits, {1, 1, vocab, 1}), indices), {1, 1, 1, count});
        auto probabilities = g.node("GGML_OP_SOFT_MAX", {sorted});
        auto cumulative = g.node("GGML_OP_CUMSUM", {probabilities});
        if (top_p < 1) {
            auto before = scale(cumulative, -1, float(top_p));
            auto keep = g.node("GGML_UNARY_OP_STEP", {before});
            auto crossing = g.node("GGML_OP_SUM_ROWS", {keep});
            crossing = g.node("GGML_OP_CLAMP", {crossing}, 0, {{"clamp_min", 0.f}, {"clamp_max", float(count - 1)}});
            std::vector<float> offsets(static_cast<size_t>(count));
            for (int64_t i = 0; i < count; ++i)
                offsets[size_t(i)] = float(i);
            keep = g.node(
                "GGML_UNARY_OP_STEP",
                {add(g.node("GGML_OP_SUB", {crossing, constant(offsets, {1, 1, 1, size_t(count)})}), scalar(.5f))});
            sorted = add(sorted, g.node("GGML_OP_LOG", {keep}));
            cumulative = g.node("GGML_OP_CUMSUM", {g.node("GGML_OP_SOFT_MAX", {sorted})});
        }
        cumulative = g.node("GGML_OP_DIV", {cumulative, slice(cumulative, 3, count - 1, 1)});
        auto crosses = g.node("GGML_UNARY_OP_STEP", {g.node("GGML_OP_SUB", {cumulative, random})});
        auto index = cast(scale(g.node("GGML_OP_SUM_ROWS", {crosses}), -1, float(count)), ov::element::i32);
        return reshape(rows(reshape(indices, {1, 1, count, 1}), index), {1, 1, 1, 1});
    }

    void code_predictor() {
        const std::string root = "a.gen.code.";
        const auto heads = g.metadata().get_int("clip.gen.audio.attention.head_count").value_or(0);
        const auto kv_heads = g.metadata().get_int("clip.gen.audio.attention.head_count_kv").value_or(heads);
        const auto layers = g.metadata().get_int("clip.gen.audio.block_count").value_or(0);
        const auto eps = float(g.metadata().get_float("clip.gen.audio.attention.layer_norm_epsilon").value_or(1e-6));
        OPENVINO_ASSERT(heads > 0 && kv_heads > 0 && heads % kv_heads == 0 && layers > 0 && eps > 0,
                        "[GGUF] invalid Qwen3TTS code predictor dimensions");
        auto embeddings = g.tensors().require(root + "embd.weight");
        auto output_embeddings = g.tensors().require(root + "out_embd.weight");
        auto head = g.tensors().require(root + "head.weight");
        const auto groups = head.ne(2);
        OPENVINO_ASSERT(groups > 0 && groups == embeddings.ne(2), "[GGUF] Qwen3TTS codebook count mismatch");
        const auto hidden_width = output_embeddings.ne(0);
        auto raw = g.add_input("gen_audio.hidden_states", ov::element::f32, {1, 1, 1, hidden_width});
        auto code0 = g.add_input("gen_audio.code0", ov::element::i32, {1, 1, 1, 1});
        auto feedback = reshape(rows(output_embeddings, code0), {1, 1, 1, hidden_width});
        std::vector<GgufValue> keys(static_cast<size_t>(layers)), values(static_cast<size_t>(layers));
        auto current_codes = code0;
        auto previous = code0;
        const auto project = [&](const GgufValue& value) {
            return g.tensors().has(root + "proj_in.weight") ? linear(value, root + "proj_in") : value;
        };
        for (int64_t pos = 0; pos <= groups; ++pos) {
            auto x = pos == 0 ? raw : pos == 1 ? feedback : rows(table(embeddings, pos - 2), previous);
            x = project(reshape(x, {1, 1, 1, hidden_width}));
            auto positions = cast(scalar(float(pos)), ov::element::i32);
            for (int64_t il = 0; il < layers; ++il) {
                const auto base = root + "blk." + std::to_string(il) + ".";
                auto h = norm(x, base + "ln1", eps);
                auto q = linear(h, base + "attn_q"), k = linear(h, base + "attn_k"), v = linear(h, base + "attn_v");
                const auto width = q.ne(0) / heads;
                OPENVINO_ASSERT(width > 0 && width % 2 == 0 && k.ne(0) == kv_heads * width && v.ne(0) == k.ne(0),
                                "[GGUF] Qwen3TTS code predictor attention geometry mismatch");
                q = norm(reshape(q, {1, 1, heads, width}), base + "attn_q_norm", eps);
                k = norm(reshape(k, {1, 1, kv_heads, width}), base + "attn_k_norm", eps);
                q = rope(q, positions, 1000000.f, width);
                k = rope(k, positions, 1000000.f, width);
                v = reshape(v, {1, 1, kv_heads, width});
                keys[size_t(il)] = pos ? concat(keys[size_t(il)], k, 1) : k;
                values[size_t(il)] = pos ? concat(values[size_t(il)], v, 1) : v;
                auto attn = g.node("GGML_OP_FLASH_ATTN_EXT",
                                   {q, keys[size_t(il)], values[size_t(il)]},
                                   100,
                                   {{"scale", 1.f / std::sqrt(float(width))}, {"f32_attention", true}});
                x = add(x, linear(reshape(attn, {1, 1, 1, heads * width}), base + "attn_out"));
                x = add(x, ffn(norm(x, base + "ln2", eps), base));
            }
            if (pos) {
                auto logits = g.node("GGML_OP_MUL_MAT", {table(head, pos - 1), norm(x, root + "output_norm", eps)});
                g.set_output(logits, "gen_audio.logits." + std::to_string(pos - 1));
                auto random =
                    g.add_input("gen_audio.random." + std::to_string(pos - 1), ov::element::f32, {1, 1, 1, 1});
                previous = sample(logits, random);
                current_codes = concat(current_codes, previous, 2);
                feedback = add(feedback, reshape(rows(table(embeddings, pos - 1), previous), {1, 1, 1, hidden_width}));
            }
        }
        g.set_output(current_codes, "gen_audio.codes");
        g.set_output(feedback, "gen_audio.embeddings");
    }

    GgufValue state(const std::string& name, int64_t channels, int64_t length) {
        return g.add_input("gen_audio.state_in." + name, ov::element::f32, {1, channels, 1, length});
    }
    void update(const std::string& name, const GgufValue& value) {
        g.set_output(value, "gen_audio.state_out." + name);
    }
    GgufValue causal_conv(GgufValue x,
                          const std::string& base,
                          int64_t dilation,
                          const std::string& name,
                          bool depthwise = false) {
        auto weight = g.tensors().require(base + ".weight");
        const auto kernel = weight.ne(0), input_channels = x.ne(2), output_channels = weight.ne(2);
        const auto pad = (kernel - 1) * dilation;
        const auto length = x.ne(0);
        if (pad) {
            x = concat(state(name, input_channels, pad), x, 3);
            update(name, slice(x, 3, length, pad));
        }
        const auto source_type = g.tensors().source_type(base + ".weight");
        const bool f16_im2col =
            !depthwise && source_type && (*source_type == GGUF_TYPE_F32 || *source_type == GGUF_TYPE_F16);
        auto w = reshape(weight, {output_channels, weight.ne(1), 1, kernel});
        auto y = g.node("GGML_OP_CONV_2D",
                        {w, x},
                        0,
                        {{"conv_params", std::vector<int64_t>{1, 1, 0, 0, dilation, 1}},
                         {"depthwise", depthwise},
                         {"im2col_precision_f16", f16_im2col}});
        if (auto bias = g.tensors()(base + ".bias"))
            y = add(y, reshape(bias, {1, output_channels, 1, 1}));
        return y;
    }
    GgufValue causal_transpose(const GgufValue& x, const std::string& base, int64_t stride, const std::string& name) {
        auto weight = g.tensors().require(base + ".weight");
        const auto channels = weight.ne(1), trim = weight.ne(0) - stride, emit = x.ne(0) * stride;
        auto y = g.node("GGML_OP_CONV_TRANSPOSE_1D",
                        {reshape(weight, {weight.ne(2), channels, weight.ne(0)}), reshape(x, {1, x.ne(2), x.ne(0)})},
                        0,
                        {{"stride", stride}});
        y = reshape(y, {1, channels, 1, emit + trim});
        if (trim) {
            OPENVINO_ASSERT(emit >= trim, "[GGUF] Qwen3TTS waveform chunk too short");
            auto head = add(slice(y, 3, 0, trim), state(name, channels, trim));
            update(name, slice(y, 3, emit, trim));
            y = emit == trim ? head : concat(head, slice(y, 3, trim, emit - trim), 3);
        }
        if (auto bias = g.tensors()(base + ".bias"))
            y = add(y, reshape(bias, {1, channels, 1, 1}));
        return y;
    }
    GgufValue snake(const GgufValue& x, const std::string& base) {
        auto alpha = reshape(g.tensors().require(base + ".alpha"), {1, x.ne(2), 1, 1});
        auto beta = reshape(g.tensors().require(base + ".beta"), {1, x.ne(2), 1, 1});
        return add(x, mul(g.node("GGML_OP_SQR", {g.node("GGML_OP_SIN", {mul(x, alpha)})}), beta));
    }
    GgufValue channels_last(const GgufValue& x) {
        return transpose(x, {0, 2, 3, 1});
    }
    GgufValue channels_first(const GgufValue& x) {
        return transpose(x, {0, 3, 1, 2});
    }

    void waveform() {
        const std::string root = "a.gen.wav.";
        constexpr int64_t frames = 72, heads = 16, prefix = frames - 1;
        auto acoustic = g.tensors().require(root + "quant.rest.codebook.weight");
        const auto groups = acoustic.ne(2);
        auto codes = g.add_input("gen_audio.audio_codes", ov::element::i32, {1, 1, groups + 1, frames});
        auto semantic = rows(g.tensors().require(root + "quant.first.codebook.weight"), slice(codes, 2, 0, 1));
        auto hidden = linear(semantic, root + "quant.first.out_proj");
        GgufValue sum;
        for (int64_t group = 0; group < groups; ++group) {
            auto embd = rows(table(acoustic, group), slice(codes, 2, group + 1, 1));
            sum = sum ? add(sum, embd) : embd;
        }
        hidden = add(hidden, linear(sum, root + "quant.rest.out_proj"));
        auto x = causal_conv(channels_first(hidden), root + "pre_conv", 1, "pre_conv");
        auto cur = linear(channels_last(x), root + "tfm.in_proj");
        auto position_state = state("tfm_pos", 1, 1);
        std::vector<float> offsets(static_cast<size_t>(frames)), key_offsets(static_cast<size_t>(prefix + frames));
        for (int64_t i = 0; i < frames; ++i)
            offsets[size_t(i)] = float(i);
        for (int64_t i = 0; i < prefix + frames; ++i)
            key_offsets[size_t(i)] = float(i);
        auto positions = cast(add(position_state, constant(offsets, {1, 1, 1, size_t(frames)})), ov::element::i32);
        update("tfm_pos", scale(position_state, 1, float(frames)));
        std::vector<float> band(size_t(frames * (prefix + frames)));
        for (int64_t i = 0; i < frames; ++i)
            for (int64_t j = 0; j < prefix + frames; ++j)
                band[size_t(i * (prefix + frames) + j)] = j <= prefix + i && j > i - 1 ? 1.f : 0.f;
        auto warm = g.node("GGML_UNARY_OP_STEP",
                           {scale(add(constant(key_offsets, {1, 1, 1, size_t(prefix + frames)}), position_state),
                                  1,
                                  .5f - float(prefix))});
        auto mask = g.node("GGML_OP_LOG", {mul(warm, constant(band, {1, 1, size_t(frames), size_t(prefix + frames)}))});
        for (int64_t il = 0; il < 8; ++il) {
            const auto base = root + "tfm.blk." + std::to_string(il) + ".";
            auto h = norm(cur, base + "ln1", 1e-5f);
            auto q = linear(h, base + "attn_q"), k = linear(h, base + "attn_k"), v = linear(h, base + "attn_v");
            const auto width = q.ne(0) / heads;
            OPENVINO_ASSERT(width > 0 && width % 2 == 0 && k.ne(0) == q.ne(0) && v.ne(0) == q.ne(0),
                            "[GGUF] invalid Qwen3TTS waveform transformer head geometry");
            q = rope(reshape(q, {1, frames, heads, width}), positions, 10000.f, width);
            k = rope(reshape(k, {1, frames, heads, width}), positions, 10000.f, width);
            v = reshape(v, {1, frames, heads, width});
            const auto cache = [&](const GgufValue& value, const std::string& name) {
                auto old = reshape(state(name, prefix, heads * width), {1, prefix, heads, width});
                auto full = concat(old, value, 1);
                update(name, reshape(slice(full, 1, frames, prefix), {1, prefix, 1, heads * width}));
                return full;
            };
            auto full_k = cache(k, "tfm_k_" + std::to_string(il));
            auto full_v = cache(v, "tfm_v_" + std::to_string(il));
            auto attn = g.node("GGML_OP_FLASH_ATTN_EXT",
                               {q, full_k, full_v, mask},
                               100,
                               {{"scale", 1.f / std::sqrt(float(width))}, {"f32_attention", true}});
            auto out = linear(reshape(attn, {1, 1, frames, heads * width}), base + "attn_out");
            if (auto scale_weight = g.tensors()(base + "ls1.weight"))
                out = mul(out, scale_weight);
            cur = add(cur, out);
            out = ffn(norm(cur, base + "ln2", 1e-5f), base);
            if (auto scale_weight = g.tensors()(base + "ls2.weight"))
                out = mul(out, scale_weight);
            cur = add(cur, out);
        }
        x = channels_first(linear(norm(cur, root + "tfm.output_norm", 1e-5f), root + "tfm.out_proj"));
        for (int64_t il = 0; il < 2; ++il) {
            const auto base = root + "up.blk." + std::to_string(il) + ".";
            x = causal_transpose(x, base + "conv", 2, "");
            auto h = channels_last(causal_conv(x, base + "dwconv", 1, "up" + std::to_string(il) + "_dwconv", true));
            h = g.build_norm_ln(h,
                                g.tensors().require(base + "norm.weight"),
                                g.tensors().require(base + "norm.bias"),
                                1e-6f);
            h = linear(g.node("GGML_UNARY_OP_GELU", {linear(h, base + "pw1")}), base + "pw2");
            x = add(x, channels_first(mul(h, g.tensors().require(base + "gamma"))));
        }
        x = causal_conv(x, root + "dac.entry", 1, "dac_entry");
        const int64_t dilations[] = {1, 3, 9};
        for (int64_t il = 0; il < 4; ++il) {
            const auto base = root + "dac.blk." + std::to_string(il) + ".";
            const auto name = "dac" + std::to_string(il);
            const auto stride = g.tensors().require(base + "conv.weight").ne(0) / 2;
            x = causal_transpose(snake(x, base + "snake"), base + "conv", stride, name + "_tail");
            for (int64_t ir = 0; ir < 3; ++ir) {
                const auto block = base + "res." + std::to_string(ir) + ".";
                auto h = causal_conv(snake(x, block + "act1"),
                                     block + "conv1",
                                     dilations[ir],
                                     name + "_res" + std::to_string(ir));
                h = causal_conv(snake(h, block + "act2"), block + "conv2", 1, "");
                x = add(x, h);
            }
        }
        x = causal_conv(snake(x, root + "dac.post_snake"), root + "dac.post_conv", 1, "dac_post_conv");
        x = g.node("GGML_OP_CLAMP", {x}, 0, {{"clamp_min", -1.f}, {"clamp_max", 1.f}});
        g.set_output(reshape(x, {-1}), "gen_audio.waveform");
    }
};

}  // namespace ov::frontend::gguf
