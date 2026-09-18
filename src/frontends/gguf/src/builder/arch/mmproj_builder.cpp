// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mmproj_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>

#include "builder/api/metadata_store.hpp"
#include "builder/gguf_graph.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov::frontend::gguf {
namespace {

enum class EncoderTopology { Siglip, Clip, Whisper, Qwen, Internvl };

struct ProjectorDefinition {
    const char* modality;
    const char* name;
    EncoderTopology topology;
};

// Entries describe implemented graph topologies, independently of language DecoderConfig.
constexpr ProjectorDefinition projector_catalog[] = {
    {"vision", "gemma3", EncoderTopology::Siglip},
    {"vision", "idefics3", EncoderTopology::Siglip},
    {"vision", "janus_pro", EncoderTopology::Siglip},
    {"vision", "mlp", EncoderTopology::Clip},
    {"vision", "internvl", EncoderTopology::Internvl},
    {"vision", "qwen2vl_merger", EncoderTopology::Qwen},
    {"vision", "qwen2.5vl_merger", EncoderTopology::Qwen},
    {"vision", "qwen3vl_merger", EncoderTopology::Qwen},
    {"audio", "qwen2a", EncoderTopology::Whisper},
    {"audio", "ultravox", EncoderTopology::Whisper},
    {"audio", "voxtral", EncoderTopology::Whisper},
    {"audio", "musicflamingo", EncoderTopology::Whisper},
    {"audio", "meralion", EncoderTopology::Whisper},
    {"audio", "glma", EncoderTopology::Whisper},
};

struct EncoderConfig {
    std::string modality, projector, prefix, activation;
    int64_t width, heads, layers, image_size = 0, patch = 0, merge = 1;
    EncoderTopology topology;
    int64_t window_pattern = 0;
    std::vector<int64_t> feature_layers;
    float eps;
};

int64_t positive(const GgufMetadata& meta, const std::string& key) {
    const auto value = meta.get_int(key);
    OPENVINO_ASSERT(value && *value > 0 && *value <= std::numeric_limits<int>::max(),
                    "[GGUF] mmproj requires a positive integer '",
                    key,
                    "'");
    return *value;
}

EncoderConfig config(const GgufMetadata& meta, const std::string& modality) {
    EncoderConfig c;
    c.modality = modality;
    c.prefix = modality == "vision" ? "v." : "a.";
    const auto key = "clip." + modality + ".";
    c.projector = meta.get_str("clip.projector_type").value_or("");
    if (c.projector.empty())
        c.projector = meta.get_str(key + "projector_type").value_or("");
    if (c.projector == "qwen2.5o")
        c.projector = modality == "vision" ? "qwen2.5vl_merger" : "qwen2a";
    const auto entry =
        std::find_if(std::begin(projector_catalog), std::end(projector_catalog), [&](const auto& candidate) {
            return modality == candidate.modality && c.projector == candidate.name;
        });
    OPENVINO_ASSERT(entry != std::end(projector_catalog),
                    "[GGUF] unsupported ",
                    modality,
                    " mmproj projector '",
                    c.projector,
                    "'");
    c.topology = entry->topology;
    c.width = positive(meta, key + "embedding_length");
    c.heads = positive(meta, key + "attention.head_count");
    c.layers = positive(meta, key + "block_count");
    if (c.topology == EncoderTopology::Clip) {
        c.feature_layers = meta.get_int_array(key + "feature_layer");
        for (auto layer : c.feature_layers)
            OPENVINO_ASSERT(layer >= 0 && layer <= c.layers, "[GGUF] invalid CLIP feature layer ", layer);
        c.layers = c.feature_layers.empty() ? c.layers - 1
                                            : *std::max_element(c.feature_layers.begin(), c.feature_layers.end());
    }
    OPENVINO_ASSERT(c.width % c.heads == 0, "[GGUF] mmproj embedding width must be divisible by head count");
    const auto eps = meta.get_float(key + "attention.layer_norm_epsilon");
    OPENVINO_ASSERT(eps && std::isfinite(*eps) && *eps > 0, "[GGUF] invalid ", key, "attention.layer_norm_epsilon");
    c.eps = static_cast<float>(*eps);
    const bool gelu = meta.get_bool("clip.use_gelu").value_or(false);
    const bool silu = meta.get_bool("clip.use_silu").value_or(false);
    OPENVINO_ASSERT(!(gelu && silu), "[GGUF] mmproj cannot enable both GELU and SiLU");
    c.activation = modality == "audio" ? "GGML_UNARY_OP_GELU_ERF"
                   : gelu              ? "GGML_UNARY_OP_GELU"
                                       : "GGML_UNARY_OP_GELU_QUICK";
    if (modality == "vision" && silu)
        c.activation = "GGML_UNARY_OP_SILU";
    if (modality == "vision") {
        c.patch = positive(meta, key + "patch_size");
        if (c.topology == EncoderTopology::Qwen) {
            c.merge = meta.get_int(key + "spatial_merge_size").value_or(2);
            c.window_pattern = c.projector == "qwen2.5vl_merger" ? positive(meta, key + "n_wa_pattern") : 0;
            OPENVINO_ASSERT(c.merge == 2 && c.width / c.heads % 4 == 0,
                            "[GGUF] Qwen vision requires merge size two and head width divisible by four");
            return c;
        }
        c.image_size = positive(meta, key + "image_size");
        c.merge = c.projector == "janus_pro" || c.topology == EncoderTopology::Clip
                      ? 1
                      : meta.get_int(key + "projector.scale_factor").value_or(c.projector == "gemma3" ? 4 : 2);
        OPENVINO_ASSERT(c.merge > 0 && c.image_size % c.patch == 0 && (c.image_size / c.patch) % c.merge == 0,
                        "[GGUF] incompatible mmproj patch/merge dimensions");
    }
    return c;
}

class MmprojBuilder : public ModelBuilder {
public:
    explicit MmprojBuilder(const BuildContext& context) : ctx(context), g(context) {}

    std::shared_ptr<GgufGraph> build() override {
        std::vector<EncoderConfig> encoders;
        for (const auto* modality : {"vision", "audio"}) {
            if (ctx.metadata.get_bool(std::string("clip.has_") + modality + "_encoder").value_or(false))
                encoders.push_back(config(ctx.metadata, modality));
        }
        OPENVINO_ASSERT(!encoders.empty(), "[GGUF] mmproj has no encoder");
        for (const auto& c : encoders) {
            auto output = c.modality == "vision" ? vision(c) : audio(c);
            g.set_output(output, c.modality + ".embeddings");
        }
        auto graph = g.finish();
        // Preserve full metadata names. Strings serialize through the standard IR rt_info path.
        for (const auto& [key, value] : detail::MetadataAccess::get(ctx.metadata).map) {
            if (key.rfind("clip.", 0) != 0)
                continue;
            if (const auto s = ctx.metadata.get_str(key)) {
                graph->mmproj_config[key] = *s;
            } else if (const auto n = ctx.metadata.get_int(key)) {
                graph->mmproj_config[key] = std::to_string(*n);
            } else if (const auto f = ctx.metadata.get_float(key)) {
                std::ostringstream stream;
                stream.precision(std::numeric_limits<double>::max_digits10);
                stream << *f;
                graph->mmproj_config[key] = stream.str();
            } else if (std::holds_alternative<std::vector<std::string>>(value)) {
                // Length-prefixed strings preserve commas, quotes and empty entries.
                std::ostringstream stream;
                for (const auto& item : ctx.metadata.get_str_array(key))
                    stream << item.size() << ':' << item;
                graph->mmproj_config[key] = stream.str();
                graph->mmproj_config[key + ".encoding"] = std::string("length-prefixed-strings");
            } else if (const auto integers = ctx.metadata.get_int_array(key); !integers.empty()) {
                std::ostringstream stream;
                for (size_t i = 0; i < integers.size(); ++i)
                    stream << (i ? "," : "") << integers[i];
                graph->mmproj_config[key] = stream.str();
            } else {
                const auto numbers = ctx.metadata.get_float_array(key);
                std::ostringstream stream;
                stream.precision(std::numeric_limits<double>::max_digits10);
                for (size_t i = 0; i < numbers.size(); ++i)
                    stream << (i ? "," : "") << numbers[i];
                graph->mmproj_config[key] = stream.str();
            }
        }
        graph->mmproj_config["version"] = std::string("1");
        for (const auto& c : encoders) {
            graph->mmproj_config[c.modality + ".projector"] = c.projector;
            graph->mmproj_config[c.modality + ".output"] = c.modality + ".embeddings";
            graph->mmproj_config[c.modality + ".output_layout"] = std::string("1,B,T,D");
            graph->mmproj_config[c.modality + ".merge"] = std::to_string(c.merge);
        }
        graph->mmproj_config["vision.auxiliary_count"] = std::to_string(auxiliary.size());
        if (!auxiliary.empty()) {
            std::string roles = "image_features";
            for (size_t i = 0; i < auxiliary.size(); ++i)
                roles += ",deepstack_features." + std::to_string(i);
            graph->mmproj_config["vision.output_roles"] = roles;
        }
        return graph;
    }

private:
    BuildContext ctx;
    GgufGraphContext g;
    std::vector<GgufValue> auxiliary;

    GgufValue reshape(const GgufValue& x, std::vector<int64_t> shape) {
        return g.node("GGML_OP_RESHAPE", {x}, 0, {{"reshape_target", shape}});
    }
    GgufValue transpose(const GgufValue& x, std::vector<int64_t> perm = {0, 1, 3, 2}) {
        return g.node("GGML_OP_TRANSPOSE", {x}, 0, {{"perm", perm}});
    }
    GgufValue add(const GgufValue& x, const GgufValue& y) {
        return g.node("GGML_OP_ADD", {x, y});
    }
    GgufValue mul(const GgufValue& x, const GgufValue& y) {
        return g.node("GGML_OP_MUL", {x, y});
    }
    GgufValue linear(const GgufValue& x, const std::string& base) {
        auto y = g.node("GGML_OP_MUL_MAT", {g.tensors().require(base + ".weight"), x});
        if (auto bias = g.tensors()(base + ".bias"))
            y = add(y, bias);
        return y;
    }
    GgufValue norm(const GgufValue& x, const std::string& base, float eps) {
        return g.build_norm_ln(x, g.tensors()(base + ".weight"), g.tensors()(base + ".bias"), eps);
    }
    GgufValue encoder_norm(const GgufValue& x, const std::string& base, const EncoderConfig& c) {
        if (c.projector != "qwen2.5vl_merger" &&
            !(c.topology == EncoderTopology::Internvl && c.width == 3200 && c.layers == 45))
            return norm(x, base, c.eps);
        auto y = g.build_norm(x, g.tensors()(base + ".weight"), c.eps);
        if (auto bias = g.tensors()(base + ".bias"))
            y = add(y, bias);
        return y;
    }
    GgufValue ffn(const GgufValue& x,
                  const std::string& up,
                  const std::string& down,
                  const std::string& activation,
                  const std::string& gate = "") {
        auto y = linear(x, up);
        if (!gate.empty() && g.tensors().has(gate + ".weight"))
            y = mul(y, g.node(activation, {linear(x, gate)}));
        else
            y = g.node(activation, {y});
        return linear(y, down);
    }
    GgufValue pool(const GgufValue& x, int64_t kx, int64_t ky) {
        return g.node("GGML_OP_POOL_2D", {x}, 0, {{"pool_params", std::vector<int64_t>{1, kx, ky, kx, ky, 0, 0}}});
    }
    GgufValue vit(GgufValue x,
                  const EncoderConfig& c,
                  const GgufValue& positions,
                  const GgufValue& rope_positions = {},
                  const GgufValue& window_mask = {}) {
        if (positions)
            x = add(x, positions);
        if (g.tensors().has(c.prefix + "pre_ln.weight"))
            x = encoder_norm(x, c.prefix + "pre_ln", c);
        std::vector<GgufValue> features;
        const auto save_feature = [&](int64_t layer, const GgufValue& value) {
            if (std::find(c.feature_layers.begin(), c.feature_layers.end(), layer) != c.feature_layers.end())
                features.push_back(value);
        };
        for (int64_t i = 0; i < c.layers; ++i) {
            save_feature(i, x);
            const auto p = c.prefix + "blk." + std::to_string(i) + ".";
            auto z = encoder_norm(x, p + "ln1", c);
            const bool fused = g.tensors().has(p + "attn_qkv.weight");
            auto qkv = fused ? linear(z, p + "attn_qkv") : GgufValue{};
            const auto projection = [&](const std::string& name, int64_t offset) {
                auto value = fused ? g.node("GGML_OP_VIEW",
                                            {qkv},
                                            3,
                                            {{"view_slice", std::vector<int64_t>{3, offset * c.width, c.width}}})
                                   : linear(z, p + name);
                const auto weight = g.tensors()(p + name + "_norm.weight");
                const bool per_head =
                    fused || (g.tensors().has(p + "attn_q_norm.weight") &&
                              g.tensors().require(p + "attn_q_norm.weight").ne(0) == c.width / c.heads);
                if (weight && !per_head)
                    value = g.build_norm_ln(value, weight, {}, c.eps);
                value = reshape(value, {1, -1, c.heads, c.width / c.heads});
                if (weight && per_head)
                    value = g.build_norm_ln(value, weight, {}, c.eps);
                return value;
            };
            auto q = projection("attn_q", 0);
            auto k = projection("attn_k", 1);
            auto v = projection("attn_v", 2);
            if (rope_positions) {
                RopeConfig rope;
                rope.n_dims = int(c.width / c.heads / 2);
                rope.freq_base = 10000.f;
                rope.freq_scale = rope.attn_factor = 1.f;
                rope.sections.fill(int32_t(c.width / c.heads / 4));
                q = g.node("GGML_OP_ROPE", {q, rope_positions}, 3 << 16, {{"rope_config", rope}});
                k = g.node("GGML_OP_ROPE", {k, rope_positions}, 3 << 16, {{"rope_config", rope}});
            }
            std::vector<GgufValue> attention_inputs{q, k, v};
            if (window_mask && (i + 1) % c.window_pattern != 0)
                attention_inputs.push_back(window_mask);
            z = g.node("GGML_OP_FLASH_ATTN_EXT",
                       attention_inputs,
                       0,
                       {{"encoder_attention", true}, {"scale", 1.f / std::sqrt(float(c.width / c.heads))}});
            z = linear(reshape(z, {1, 1, -1, c.width}), p + "attn_out");
            if (auto scale = g.tensors()(p + "ls1.weight"))
                z = mul(z, scale);
            if (auto weight = g.tensors()(p + "attn_post_norm.weight"))
                z = g.build_norm_ln(z, weight, {}, c.eps);
            x = add(x, z);
            const bool legacy_swap =
                (c.projector == "gemma3" || c.projector == "idefics3" || c.topology == EncoderTopology::Clip ||
                 c.projector == "qwen2vl_merger" || c.projector == "qwen2.5vl_merger") &&
                g.tensors().require(p + "ffn_down.weight").ne(0) == c.width;
            z = ffn(encoder_norm(x, p + "ln2", c),
                    p + (legacy_swap ? "ffn_down" : "ffn_up"),
                    p + (legacy_swap ? "ffn_up" : "ffn_down"),
                    c.activation,
                    p + "ffn_gate");
            if (auto weight = g.tensors()(p + "ffn_post_norm.weight"))
                z = g.build_norm_ln(z, weight, {}, c.eps);
            if (auto scale = g.tensors()(p + "ls2.weight"))
                z = mul(z, scale);
            x = add(x, z);
            if (auto scale = g.tensors()(p + "out_scale.weight"))
                x = mul(x, scale);
            const auto deep = "v.deepstack." + std::to_string(i);
            if (c.projector == "qwen3vl_merger" && g.tensors().has(deep + ".norm.weight")) {
                auto feature = norm(reshape(x, {1, 1, -1, 4 * c.width}), deep + ".norm", c.eps);
                auxiliary.push_back(ffn(feature, deep + ".fc1", deep + ".fc2", "GGML_UNARY_OP_GELU"));
            }
        }
        if (c.projector == "qwen2a" || c.projector == "voxtral" || c.projector == "musicflamingo")
            x = transpose(pool(transpose(x), 2, 1));
        if (g.tensors().has(c.prefix + "post_ln.weight"))
            x = encoder_norm(x, c.prefix + "post_ln", c);
        save_feature(c.layers, x);
        if (!features.empty()) {
            x = features.front();
            for (size_t i = 1; i < features.size(); ++i)
                x = g.node("GGML_OP_CONCAT", {x, features[i]}, 0, {{"concat_axis", 0}});
        }
        return x;
    }
    GgufValue vision(const EncoderConfig& c) {
        if (c.topology == EncoderTopology::Qwen)
            return qwen_vision(c);
        auto x = g.add_input("vision.pixel_values", ov::element::f32, {1, 3, c.image_size, c.image_size});
        const auto w = g.tensors().require("v.patch_embd.weight");
        x = g.node("GGML_OP_CONV_2D", {w, x}, 0, {{"conv_params", std::vector<int64_t>{c.patch, c.patch, 0, 0, 1, 1}}});
        x = transpose(reshape(x, {1, 1, c.width, -1}));
        if (auto bias = g.tensors()("v.patch_embd.bias"))
            x = add(x, bias);
        if ((c.topology == EncoderTopology::Clip || c.topology == EncoderTopology::Internvl) &&
            g.tensors().has("v.class_embd"))
            x = g.node("GGML_OP_CONCAT",
                       {x, reshape(g.tensors().require("v.class_embd"), {1, 1, 1, c.width})},
                       0,
                       {{"concat_axis", 1}});
        x = vit(x, c, g.tensors().require("v.position_embd.weight"));
        const auto side = c.image_size / c.patch;
        if (c.topology == EncoderTopology::Internvl) {
            OPENVINO_ASSERT(g.tensors().has("v.class_embd"), "[GGUF] InternVL requires a class embedding");
            x = g.node("GGML_OP_VIEW", {x}, 3, {{"view_slice", std::vector<int64_t>{2, 0, side * side}}});
            x = reshape(x, {1, side, side / c.merge, c.width * c.merge});
            x = transpose(x, {0, 2, 1, 3});
            x = reshape(x, {1, side / c.merge, side / c.merge, c.width * c.merge * c.merge});
            x = transpose(x, {0, 2, 1, 3});
            x = reshape(x, {1, 1, -1, c.width * c.merge * c.merge});
            return ffn(norm(x, "mm.model.mlp.0", 1e-5f), "mm.model.mlp.1", "mm.model.mlp.3", "GGML_UNARY_OP_GELU");
        }
        if (c.topology == EncoderTopology::Clip) {
            const int64_t offset = g.tensors().has("v.class_embd") ? 1 : 0;
            x = g.node("GGML_OP_VIEW", {x}, 3, {{"view_slice", std::vector<int64_t>{2, offset, side * side}}});
            x = linear(x, "mm.0");
            const bool normalized = g.tensors().has("mm.3.weight");
            if (normalized)
                x = norm(x, "mm.1", c.eps);
            x = g.node("GGML_UNARY_OP_GELU", {x});
            if (normalized)
                return norm(linear(x, "mm.3"), "mm.4", c.eps);
            return g.tensors().has("mm.2.weight") ? linear(x, "mm.2") : x;
        }
        if (c.projector == "gemma3") {
            x = reshape(transpose(x), {1, c.width, side, side});
            x = transpose(reshape(pool(x, c.merge, c.merge), {1, 1, c.width, -1}));
            x = g.build_norm(x, g.tensors().require("mm.soft_emb_norm.weight"), c.eps);
            return g.node("GGML_OP_MUL_MAT", {transpose(g.tensors().require("mm.input_projection.weight"), {1, 0}), x});
        }
        if (c.projector == "idefics3") {
            // Gather each spatial merge window in row-major pixel order, then project it.
            std::vector<int32_t> order;
            for (int64_t y = 0; y < side; y += c.merge)
                for (int64_t xx = 0; xx < side; xx += c.merge)
                    for (int64_t dy = 0; dy < c.merge; ++dy)
                        for (int64_t dx = 0; dx < c.merge; ++dx)
                            order.push_back(int32_t((y + dy) * side + xx + dx));
            ov::Tensor indices(ov::element::i32, {1, 1, 1, order.size()});
            std::copy(order.begin(), order.end(), indices.data<int32_t>());
            x = g.node("GGML_OP_GET_ROWS", {x, g.add_constant("vision.merge_indices", indices)});
            return linear(reshape(x, {1, 1, -1, c.width * c.merge * c.merge}), "mm.model.fc");
        }
        return ffn(x, "mm.0", "mm.1", c.activation);
    }
    GgufValue qwen_vision(const EncoderConfig& c) {
        // A temporal pair; still-image callers duplicate the image. Index inputs specify
        // spatial 2x2 grouping and, for Qwen2.5, the reference's window permutation.
        auto pixels = g.add_input("vision.pixel_values", ov::element::f32, {2, 3, -1, -1});
        const auto convolution = [&](int64_t frame, const std::string& name) {
            auto image = g.node("GGML_OP_VIEW", {pixels}, 3, {{"view_slice", std::vector<int64_t>{0, frame, 1}}});
            return g.node("GGML_OP_CONV_2D",
                          {g.tensors().require(name), image},
                          0,
                          {{"conv_params", std::vector<int64_t>{c.patch, c.patch, 0, 0, 1, 1}}});
        };
        auto patches = add(convolution(0, "v.patch_embd.weight"), convolution(1, "v.patch_embd.weight.1"));
        auto indices = g.add_input("vision.patch_indices", ov::element::i32, {1, 1, 1, -1});
        auto group = [&](const GgufValue& value) {
            return g.node("GGML_OP_GET_ROWS", {transpose(reshape(value, {1, 1, c.width, -1})), indices});
        };
        auto x = group(patches);
        if (auto bias = g.tensors()("v.patch_embd.bias"))
            x = add(x, bias);
        GgufValue learned_positions;
        if (c.projector == "qwen3vl_merger") {
            auto table = g.tensors().require("v.position_embd.weight");
            const int64_t side = int64_t(std::sqrt(double(table.ne(1))));
            OPENVINO_ASSERT(side * side == table.ne(1), "[GGUF] Qwen position table must have a square grid");
            auto grid = transpose(reshape(table, {1, side, side, c.width}), {0, 3, 1, 2});
            auto resized = g.node("GGML_OP_UPSCALE",
                                  {grid, patches},
                                  0,
                                  {{"resize_like", true}, {"interpolation_mode", 1 | 0x200}});
            learned_positions = group(resized);
        }
        auto positions = g.add_input("vision.position_ids", ov::element::i32, {1, 1, 1, -1});
        GgufValue window_mask;
        if (c.window_pattern)
            window_mask = g.add_input("vision.attention_mask", ov::element::f32, {1, 1, -1, -1});
        x = vit(x, c, learned_positions, positions, window_mask);
        x = ffn(reshape(x, {1, 1, -1, 4 * c.width}), "mm.0", "mm.2", "GGML_UNARY_OP_GELU");
        for (const auto& feature : auxiliary)
            x = g.node("GGML_OP_CONCAT", {x, feature}, 0, {{"concat_axis", 0}});
        if (c.window_pattern) {
            auto order = g.add_input("vision.output_indices", ov::element::i32, {1, 1, 1, -1});
            x = g.node("GGML_OP_GET_ROWS", {x, order});
        }
        return x;
    }
    GgufValue audio(const EncoderConfig& c) {
        const auto mel = positive(ctx.metadata, "clip.audio.num_mel_bins");
        auto x = g.add_input("audio.features", ov::element::f32, {1, mel, 1, -1});
        for (int i = 1; i <= 2; ++i) {
            const auto base = "a.conv1d." + std::to_string(i);
            auto w = g.tensors().require(base + ".weight");
            const auto kernel = w.ne(0), in = w.ne(1), out = w.ne(2);
            w = reshape(w, {out, in, 1, kernel});
            x = g.node("GGML_OP_CONV_2D",
                       {w, x},
                       0,
                       {{"conv_params", std::vector<int64_t>{i, 1, kernel / 2, 0, 1, 1}}});
            x = add(x, reshape(g.tensors().require(base + ".bias"), {1, out, 1, 1}));
            x = g.node("GGML_UNARY_OP_GELU_ERF", {x});
        }
        x = transpose(reshape(x, {1, 1, c.width, -1}));
        auto ids = g.add_input("audio.position_ids", ov::element::i32, {1, 1, 1, -1});
        auto pos = g.node("GGML_OP_GET_ROWS", {g.tensors().require("a.position_embd.weight"), ids});
        x = vit(x, c, pos);
        if (c.projector == "qwen2a")
            return linear(x, "mm.a.fc");
        if (c.projector == "glma")
            x = norm(x, "mm.a.norm_pre", c.eps);
        if (c.projector == "ultravox" || c.projector == "voxtral" || c.projector == "meralion" ||
            c.projector == "glma") {
            const auto stack = positive(ctx.metadata, "clip.audio.projector.stack_factor");
            x = g.node("GGML_OP_PAD", {x}, 0, {{"pad_tokens_to_multiple", stack}});
            x = reshape(x, {1, 1, -1, c.width * stack});
        }
        if (c.projector == "ultravox") {
            x = g.build_norm(x, g.tensors().require("mm.a.norm_pre.weight"), 1e-6f);
            x = linear(x, "mm.a.mlp.1");
            x = g.node("GGML_GLU_OP_SWIGLU", {x}, 0, {{"swapped", true}});
            x = g.build_norm(x, g.tensors().require("mm.a.norm_mid.weight"), 1e-6f);
            return linear(x, "mm.a.mlp.2");
        }
        if (c.projector == "meralion") {
            x = g.node("GGML_UNARY_OP_SILU", {linear(norm(x, "mm.a.norm_pre", c.eps), "mm.a.mlp.0")});
            x = mul(g.node("GGML_UNARY_OP_SILU", {linear(x, "mm.a.mlp.1")}), linear(x, "mm.a.mlp.2"));
            return linear(x, "mm.a.mlp.3");
        }
        if (c.projector == "glma") {
            x = ffn(x, "mm.a.mlp.1", "mm.a.mlp.2", c.activation);
            const auto width = x.ne(0);
            x = g.node("GGML_OP_CONCAT",
                       {reshape(g.tensors().require("v.boi"), {1, 1, 1, width}), x},
                       0,
                       {{"concat_axis", 1}});
            return g.node("GGML_OP_CONCAT",
                          {x, reshape(g.tensors().require("v.eoi"), {1, 1, 1, width})},
                          0,
                          {{"concat_axis", 1}});
        }
        return ffn(x, "mm.a.mlp.1", "mm.a.mlp.2", "GGML_UNARY_OP_GELU_ERF");
    }
};

}  // namespace

ArchitectureDefinition mmproj_architecture() {
    return {"clip.mmproj",
            "clip",
            [](const BuildContext& context) {
                return std::make_shared<MmprojBuilder>(context);
            },
            [](const GgufMetadata& meta) {
                return meta.has("clip.projector_type") || meta.has("clip.vision.projector_type") ||
                       meta.has("clip.audio.projector_type");
            },
            Maturity::Experimental};
}

}  // namespace ov::frontend::gguf
