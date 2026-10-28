// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lfm3_projector_builder.hpp"

#include <cmath>
#include <limits>

#include "builder/metadata_checks.hpp"

namespace ov::frontend::gguf {
namespace {
class Lfm3Projector {
public:
    explicit Lfm3Projector(GgufGraphContext& graph, const std::string& modality) : g(graph) {
        const auto key = "clip." + modality + ".";
        width = require_positive_int(g.metadata(), key + "embedding_length", "LFM3 mmproj");
        heads = require_positive_int(g.metadata(), key + "attention.head_count", "LFM3 mmproj");
        kv_heads = require_positive_int(g.metadata(), key + "attention.head_count_kv", "LFM3 mmproj");
        layers = require_positive_int(g.metadata(), key + "block_count", "LFM3 mmproj");
        eps = require_positive_float(g.metadata(), key + "attention.layer_norm_epsilon", "LFM3 mmproj");
        OPENVINO_ASSERT(width % heads == 0 && heads % kv_heads == 0, "[GGUF] invalid LFM3 encoder head geometry");
        head = width / heads;
        RopeConfig config;
        config.per_op = true;
        config.n_dims = int(head);
        config.freq_base = 1e6f;
        config.freq_scale = config.attn_factor = 1.f;
        g.configure_rope(config);
    }

    ProjectorResult vision() {
        const auto patch = require_positive_int(g.metadata(), "clip.vision.patch_size", "LFM3-VL");
        const auto merge = g.metadata().get_int("clip.vision.spatial_merge_size").value_or(2);
        OPENVINO_ASSERT(patch == 16 && merge == 2 && head % 4 == 0, "[GGUF] unsupported LFM3-VL patch/merge geometry");
        auto pixels = g.add_input("vision.pixel_values", ov::element::f32, {1, 3, -1, -1});
        auto spatial = conv(pixels, "v.patch_embd", patch, patch, 0, 0);
        if (g.tensors().has("v.patch_embd.weight.1")) {
            auto second = g.add_input("vision.pixel_values_1", ov::element::f32, {1, 3, -1, -1});
            spatial = add(spatial,
                          g.node("GGML_OP_CONV_2D",
                                 {g.tensors().require("v.patch_embd.weight.1"), second},
                                 0,
                                 {{"conv_params", std::vector<int64_t>{patch, patch, 0, 0, 1, 1}}}));
        }
        auto x = transpose(reshape(spatial, {1, 1, width, -1}));
        x = rms(x, "v.embedding_norm");
        auto positions = g.add_input("vision.positions", ov::element::i32, {1, 1, 1, -1});
        for (int64_t layer = 0; layer < layers; ++layer) {
            const auto p = "v.blk." + std::to_string(layer) + ".";
            auto z = rms(x, p + "ln1");
            if (g.tensors().has(p + "conv2d_in_proj.weight")) {
                auto projection = linear(z, p + "conv2d_in_proj");
                auto gated = mul(slice(projection, 3, 0, width), slice(projection, 3, 2 * width, width));
                if (g.tensors().has(p + "conv2d_spatial.weight")) {
                    auto grid = g.node("GGML_OP_RESHAPE",
                                       {transpose(gated), spatial},
                                       0,
                                       {{"reshape_target", std::vector<int64_t>{1, width, 0, 0}},
                                        {"shape_axes", std::vector<int64_t>{-1, -1, 2, 3}}});
                    gated = transpose(reshape(conv(grid, p + "conv2d_spatial", 1, 1, 1, 1, true), {1, 1, width, -1}));
                }
                z = linear(mul(slice(projection, 3, width, width), gated), p + "conv2d_out_proj");
            } else {
                z = attention(z, p, positions, true);
            }
            x = add(x, z);
            x = add(x, ffn(rms(x, p + "ln2"), p, "GGML_UNARY_OP_SILU"));
        }
        x = g.build_norm_ln(x,
                            g.tensors().require("mm.input_norm.weight"),
                            g.tensors().require("mm.input_norm.bias"),
                            1e-6f);
        auto grid = g.node("GGML_OP_RESHAPE",
                           {transpose(x), spatial},
                           0,
                           {{"reshape_target", std::vector<int64_t>{1, width, 0, 0}},
                            {"shape_axes", std::vector<int64_t>{-1, -1, 2, 3}}});
        // Unfold is channel-major; restore raster 2x2 patches with features innermost.
        x = g.node("GGML_OP_IM2COL",
                   {grid},
                   0,
                   {{"kernel_shape", ov::Shape{size_t(4 * width), size_t(width), 2, 2}},
                    {"im2col_params", std::vector<int32_t>{2, 2, 0, 0, 1, 1, 1}},
                    {"dst_type", ov::element::f32}});
        x = reshape(transpose(reshape(x, {1, -1, width, 4}), {0, 1, 3, 2}), {1, 1, -1, 4 * width});
        x = linear(g.node("GGML_UNARY_OP_GELU_ERF", {linear(x, "mm.0")}), "mm.2");
        ProjectorResult result;
        result.output = x;
        result.genai_layout = ProjectorLayout::Image;
        result.config = {{"vision.merge", "2"}, {"vision.temporal_patch_size", "2"}, {"vision.streaming", "false"}};
        return result;
    }

    ProjectorResult audio() {
        const auto mel = require_positive_int(g.metadata(), "clip.audio.num_mel_bins", "LFM3-Audio");
        const auto window = require_positive_int(g.metadata(), "clip.audio.window_size", "LFM3-Audio");
        OPENVINO_ASSERT(mel == 128 && window > 1, "[GGUF] invalid LFM3-Audio mel/window geometry");
        prefix = window - 1;
        auto base_position = state("pos", {1, 1, 1, 1});
        auto x = g.add_input("audio.features", ov::element::f32, {1, 1, mel, -1});
        x = downsample(x, "ds0", "a.conv1d.0", false);
        x = g.node("GGML_UNARY_OP_SILU", {x});
        x = downsample(x, "ds1", "a.conv1d.2", true);
        x = g.node("GGML_UNARY_OP_SILU",
                   {add(conv(x, "a.conv1d.3", 1, 1, 0, 0), g.tensors().require("a.conv1d.3.bias"))});
        x = downsample(x, "ds2", "a.conv1d.5", true);
        x = g.node("GGML_UNARY_OP_SILU",
                   {add(conv(x, "a.conv1d.6", 1, 1, 0, 0), g.tensors().require("a.conv1d.6.bias"))});
        const auto flat = g.tensors().require("a.pre_encode.out.weight").ne(0);
        x = linear(reshape(transpose(x, {0, 3, 1, 2}), {1, 1, -1, flat}), "a.pre_encode.out");
        auto index = scale(g.node("GGML_OP_CUMSUM", {transpose(scale(slice(x, 3, 0, 1), 0.f, 1.f))}), 1.f, -1.f);
        auto positions = g.node("GGML_OP_CPY", {add(index, base_position)}, 0, {{"dst_type", ov::element::i32}});
        update(base_position, add(base_position, scale(slice(index, 3, -1, 1), 1.f, 1.f)));
        for (int64_t layer = 0; layer < layers; ++layer) {
            const auto p = "a.blk." + std::to_string(layer) + ".";
            auto z = rms(x, p + "ln1");
            if (g.tensors().has(p + "conv_dw.weight")) {
                auto projection = linear(z, p + "conv_pw1");
                auto bx = transpose(mul(slice(projection, 3, 0, width), slice(projection, 3, 2 * width, width)));
                const auto taps = g.tensors().require(p + "conv_dw.weight").ne(0) - 1;
                auto previous = state("conv_" + std::to_string(layer), {1, 1, width, taps});
                bx = concat(previous, bx, 3);
                update(previous, slice(bx, 3, -taps, taps));
                z = g.node("GGML_OP_SSM_CONV", {bx, g.tensors().require(p + "conv_dw.weight")});
                z = linear(mul(slice(projection, 3, width, width), z), p + "conv_pw2");
            } else {
                z = attention(z, p, positions, false, layer, index, base_position);
            }
            x = add(x, z);
            x = add(x, ffn(rms(x, p + "ln2"), p, "GGML_UNARY_OP_SILU"));
        }
        x = rms(x, "a.post_ln");
        x = g.build_norm_ln(x, g.tensors().require("mm.a.mlp.0.weight"), g.tensors().require("mm.a.mlp.0.bias"), 1e-5f);
        x = linear(g.node("GGML_UNARY_OP_GELU_ERF", {linear(x, "mm.a.mlp.1")}), "mm.a.mlp.3");
        ProjectorResult result;
        result.output = x;
        result.config = {{"audio.merge", "1"}, {"audio.streaming", "true"}, {"audio.encoder_stride", "8"}};
        return result;
    }

private:
    GgufGraphContext& g;
    int64_t width, heads, kv_heads, layers, head, prefix = 0;
    float eps;

    GgufValue reshape(const GgufValue& x, std::vector<int64_t> shape) {
        return g.node("GGML_OP_RESHAPE", {x}, 6, {{"reshape_target", std::move(shape)}});
    }
    GgufValue transpose(const GgufValue& x, std::vector<int64_t> perm = {0, 1, 3, 2}) {
        return g.node("GGML_OP_TRANSPOSE", {x}, 0, {{"perm", std::move(perm)}});
    }
    GgufValue slice(const GgufValue& x, int64_t axis, int64_t begin, int64_t count) {
        return g.node("GGML_OP_VIEW", {x}, 3, {{"view_slice", std::vector<int64_t>{axis, begin, count}}});
    }
    GgufValue add(const GgufValue& x, const GgufValue& y) {
        return g.node("GGML_OP_ADD", {x, y});
    }
    GgufValue mul(const GgufValue& x, const GgufValue& y) {
        return g.node("GGML_OP_MUL", {x, y});
    }
    GgufValue concat(const GgufValue& x, const GgufValue& y, int64_t axis) {
        return g.node("GGML_OP_CONCAT", {x, y}, 0, {{"concat_axis", int(3 - axis)}});
    }
    GgufValue scale(const GgufValue& x, float value, float bias = 0.f) {
        return g.node("GGML_OP_SCALE", {x}, 0, {{"scale", value}, {"bias", bias}});
    }
    GgufValue linear(const GgufValue& x, const std::string& name) {
        auto y = g.node("GGML_OP_MUL_MAT", {g.tensors().require(name + ".weight"), x});
        if (auto bias = g.tensors()(name + ".bias"))
            y = add(y, bias);
        return y;
    }
    GgufValue rms(const GgufValue& x, const std::string& name) {
        auto y = g.build_norm(x, g.tensors().require(name + ".weight"), eps);
        if (auto bias = g.tensors()(name + ".bias"))
            y = add(y, bias);
        return y;
    }
    GgufValue ffn(const GgufValue& x, const std::string& p, const std::string& activation) {
        return linear(mul(g.node(activation, {linear(x, p + "ffn_gate")}), linear(x, p + "ffn_up")), p + "ffn_down");
    }
    GgufValue
    conv(const GgufValue& x, const std::string& name, int64_t sx, int64_t sy, int64_t px, int64_t py, bool dw = false) {
        return g.node("GGML_OP_CONV_2D",
                      {g.tensors().require(name + ".weight"), x},
                      0,
                      {{"conv_params", std::vector<int64_t>{sx, sy, px, py, 1, 1}}, {"depthwise", dw}});
    }
    GgufValue state(const std::string& name, const ov::PartialShape& shape) {
        return g.add_input("audio.state." + name, ov::element::f32, shape);
    }
    void update(const GgufValue& previous, const GgufValue& next) {
        OPENVINO_ASSERT(previous.shape().compatible(next.shape()),
                        "[GGUF] LFM3 state shape mismatch ",
                        previous.name(),
                        " ",
                        previous.shape(),
                        " -> ",
                        next.shape());
        g.add_recurrent_state(previous, next);
    }
    GgufValue downsample(GgufValue x, const std::string& slot, const std::string& name, bool dw) {
        const auto shape = x.shape();
        auto previous = state(slot, {1, shape[1], shape[2], 1});
        x = concat(previous, x, 3);
        update(previous, slice(x, 3, -1, 1));
        x = g.node("GGML_OP_PAD", {x}, 0, {{"pad_params", std::vector<int32_t>{0, 1, 0, 0, 0, 0, 0, 0}}});
        return add(conv(x, name, 2, 2, 0, 1, dw), g.tensors().require(name + ".bias"));
    }
    GgufValue attention(const GgufValue& x,
                        const std::string& p,
                        const GgufValue& positions,
                        bool vision,
                        int64_t layer = 0,
                        const GgufValue& index = {},
                        const GgufValue& base_position = {}) {
        auto q = reshape(linear(x, p + "attn_q"), {1, -1, heads, head});
        auto k = reshape(linear(x, p + "attn_k"), {1, -1, kv_heads, head});
        auto v = reshape(linear(x, p + "attn_v"), {1, -1, kv_heads, head});
        OPENVINO_ASSERT(
            vision || (g.tensors().has(p + "attn_q_norm.weight") && g.tensors().has(p + "attn_k_norm.weight")),
            "[GGUF] LFM3-Audio requires per-head Q/K RMS normalization");
        if (g.tensors().has(p + "attn_q_norm.weight"))
            q = rms(q, p + "attn_q_norm");
        if (g.tensors().has(p + "attn_k_norm.weight"))
            k = rms(k, p + "attn_k_norm");
        RopeConfig rope;
        rope.per_op = true;
        rope.n_dims = int(vision ? head / 2 : head);
        rope.freq_base = 1e6f;
        rope.freq_scale = rope.attn_factor = 1.f;
        if (vision) {
            rope.n_ctx_orig = 32768;
            rope.beta_fast = 32.f;
            rope.beta_slow = 1.f;
            rope.sections = {int32_t(head / 4), int32_t(head / 4), 0, 0};
        }
        q = g.node("GGML_OP_ROPE", {q, positions}, (vision ? 3 : 1) << 16, {{"rope_config", rope}});
        k = g.node("GGML_OP_ROPE", {k, positions}, (vision ? 3 : 1) << 16, {{"rope_config", rope}});
        std::vector<GgufValue> inputs{q, k, v};
        if (!vision) {
            auto previous_k = state("k_" + std::to_string(layer), {1, 1, prefix, kv_heads * head});
            auto previous_v = state("v_" + std::to_string(layer), {1, 1, prefix, kv_heads * head});
            auto flat_k = concat(previous_k, reshape(k, {1, 1, -1, kv_heads * head}), 2);
            auto flat_v = concat(previous_v, reshape(v, {1, 1, -1, kv_heads * head}), 2);
            update(previous_k, slice(flat_k, 2, -prefix, prefix));
            update(previous_v, slice(flat_v, 2, -prefix, prefix));
            k = reshape(flat_k, {1, -1, kv_heads, head});
            v = reshape(flat_v, {1, -1, kv_heads, head});
            auto key_index = scale(g.node("GGML_OP_CUMSUM", {transpose(scale(slice(flat_k, 3, 0, 1), 0.f, 1.f))}),
                                   1.f,
                                   -1.f - float(prefix));
            auto distance = g.node("GGML_OP_SUB", {transpose(index), key_index});
            auto clamp = [&](const GgufValue& value) {
                return g.node("GGML_OP_CLAMP", {value}, 0, {{"clamp_min", 0.f}, {"clamp_max", 1.f}});
            };
            auto inside = mul(clamp(scale(distance, 1.f, 1.f)), clamp(scale(distance, -1.f, float(prefix + 1))));
            inside = mul(inside, clamp(scale(add(key_index, base_position), 1.f, 1.f)));
            inputs = {q, k, v, scale(inside, 1e9f, -1e9f)};
        }
        return linear(reshape(g.node("GGML_OP_FLASH_ATTN_EXT",
                                     inputs,
                                     100,
                                     {{"f32_attention", true}, {"scale", 1.f / std::sqrt(float(head))}}),
                              {1, 1, -1, width}),
                      p + "attn_out");
    }
};
}  // namespace

ProjectorResult build_lfm3_projector(GgufGraphContext& graph, const std::string& modality) {
    OPENVINO_ASSERT(modality == "vision" || modality == "audio", "[GGUF] invalid LFM3 projector modality");
    Lfm3Projector builder(graph, modality);
    return modality == "vision" ? builder.vision() : builder.audio();
}

std::vector<ProjectorDefinition> lfm3_projector_definitions() {
    std::vector<ProjectorDefinition> result;
    for (const auto* modality : {"vision", "audio"}) {
        const std::string type = std::string(modality) == "vision" ? "lfm3vl" : "lfm3a";
        result.push_back({"clip." + std::string(modality) + "." + type,
                          "clip",
                          modality,
                          type,
                          [modality](GgufGraphContext& graph) {
                              return build_lfm3_projector(graph, modality);
                          },
                          {}});
    }
    return result;
}
}  // namespace ov::frontend::gguf
