// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mmproj_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>

#include "builder/api/metadata_store.hpp"
#include "builder/gguf_graph.hpp"
#include "builder/projector_registry.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov::frontend::gguf {
namespace {

// Power-of-two shrink that keeps Gemma 4 unified-vision activations within F16.
constexpr float kUnifiedShrink = 1.f / 16;

enum class EncoderTopology {
    SIGLIP,
    CLIP,
    WHISPER,
    QWEN,
    INTERNVL,
    RESAMPLER,
    PIXTRAL,
    GEMMA4,
    UNIFIED_VISION,
    UNIFIED_AUDIO,
    MINICPM46,
    MUSE_GLIMMER,
    PHI4,
    GEMMA4_AUDIO,
    OCR,
    OCR2
};

// ggml rope mode bits, as passed in the op_case.
constexpr int ROPE_NEOX = 1 << 16;
constexpr int ROPE_VISION = 3 << 16;

// Per-projector traits follow llama.cpp's clip.cpp choices.
enum ProjectorTraits : unsigned {
    RMS_NORM = 1,     // encoder norms are RMS rather than layer norms
    SWAPPED_FFN = 2,  // legacy exports name ffn_up/ffn_down after each other's roles
    POOL_2 = 4,       // average-pool encoder tokens by 2 before the projector
};

struct ProjectorTopology {
    const char* modality;
    const char* name;
    EncoderTopology topology;
    unsigned traits = 0;
};

// Entries describe implemented graph topologies, independently of language DecoderConfig.
constexpr ProjectorTopology projector_catalog[] = {
    {"vision", "deepseekocr", EncoderTopology::OCR},
    {"vision", "deepseekocr2", EncoderTopology::OCR2, RMS_NORM},
    {"vision", "pixtral", EncoderTopology::PIXTRAL, RMS_NORM},
    {"vision", "phi4", EncoderTopology::PHI4},
    {"vision", "muse-glimmer", EncoderTopology::MUSE_GLIMMER},
    {"vision", "gemma4v", EncoderTopology::GEMMA4, RMS_NORM},
    {"vision", "gemma4uv", EncoderTopology::UNIFIED_VISION},
    {"audio", "gemma4a", EncoderTopology::GEMMA4_AUDIO},
    {"audio", "gemma4ua", EncoderTopology::UNIFIED_AUDIO},
    {"vision", "minicpmv4_6", EncoderTopology::MINICPM46},
    {"vision", "gemma3", EncoderTopology::SIGLIP, SWAPPED_FFN},
    {"vision", "idefics3", EncoderTopology::SIGLIP, SWAPPED_FFN},
    {"vision", "janus_pro", EncoderTopology::SIGLIP},
    {"vision", "mlp", EncoderTopology::CLIP, SWAPPED_FFN},
    {"vision", "internvl", EncoderTopology::INTERNVL},
    {"vision", "resampler", EncoderTopology::RESAMPLER},
    {"vision", "qwen2vl_merger", EncoderTopology::QWEN, SWAPPED_FFN},
    {"vision", "qwen2.5vl_merger", EncoderTopology::QWEN, RMS_NORM | SWAPPED_FFN},
    {"vision", "qwen3vl_merger", EncoderTopology::QWEN},
    {"audio", "qwen2a", EncoderTopology::WHISPER, POOL_2},
    {"audio", "ultravox", EncoderTopology::WHISPER},
    {"audio", "voxtral", EncoderTopology::WHISPER, POOL_2},
    {"audio", "musicflamingo", EncoderTopology::WHISPER, POOL_2},
    {"audio", "meralion", EncoderTopology::WHISPER},
    {"audio", "glma", EncoderTopology::WHISPER},
};

struct EncoderConfig {
    std::string projector, prefix, activation;
    int64_t width, heads, layers, image_size = 0, patch = 0, merge = 1;
    EncoderTopology topology;
    int64_t window_pattern = 0;
    int64_t version = 0, queries = 0, kv_heads = 0;
    std::vector<int64_t> feature_layers;
    float eps;
    unsigned traits = 0;
    bool rms = false;  // encoder norms are RMS rather than layer norms
};

bool rms_encoder(const EncoderConfig& c) {
    // As in llama.cpp, InternViT-6B is recognized by its geometry.
    return (c.traits & RMS_NORM) || (c.topology == EncoderTopology::INTERNVL && c.width == 3200 && c.layers == 45);
}

int64_t positive(const GgufMetadata& meta, const std::string& key) {
    const auto value = meta.get_int(key);
    OPENVINO_ASSERT(value && *value > 0 && *value <= std::numeric_limits<int>::max(),
                    "[GGUF] mmproj requires a positive integer '",
                    key,
                    "'");
    return *value;
}

EncoderConfig config(const GgufMetadata& meta, const std::string& modality, const std::string& projector) {
    EncoderConfig c;
    c.prefix = modality == "vision" ? "v." : "a.";
    const auto key = "clip." + modality + ".";
    c.projector = projector;
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
    c.traits = entry->traits;
    if (c.topology == EncoderTopology::UNIFIED_AUDIO) {
        c.width = 640;
        c.heads = 1;
        c.layers = 0;
        c.eps = 1e-6f;
        return c;
    }
    c.width = positive(meta, key + "embedding_length");
    c.heads = c.topology == EncoderTopology::UNIFIED_VISION ? 1 : positive(meta, key + "attention.head_count");
    c.kv_heads = c.topology == EncoderTopology::OCR2 ? positive(meta, key + "attention.head_count_kv") : c.heads;
    // GQA head expansion is done by the FLASH_ATTN_EXT translator.
    OPENVINO_ASSERT(c.heads % c.kv_heads == 0, "[GGUF] encoder GQA head count mismatch");
    c.layers = c.topology == EncoderTopology::UNIFIED_VISION ? 0 : positive(meta, key + "block_count");
    if (c.topology == EncoderTopology::CLIP) {
        c.feature_layers = meta.get_int_array(key + "feature_layer");
        for (auto layer : c.feature_layers)
            OPENVINO_ASSERT(layer >= 0 && layer <= c.layers, "[GGUF] invalid CLIP feature layer ", layer);
        c.layers = c.feature_layers.empty() ? c.layers - 1
                                            : *std::max_element(c.feature_layers.begin(), c.feature_layers.end());
    }
    OPENVINO_ASSERT(c.width % c.heads == 0, "[GGUF] mmproj embedding width must be divisible by head count");
    const auto eps = meta.get_float(key + "attention.layer_norm_epsilon");
    OPENVINO_ASSERT(eps && std::isfinite(*eps) && *eps > 0, "[GGUF] invalid ", key, "attention.layer_norm_epsilon");
    c.eps = c.topology == EncoderTopology::GEMMA4_AUDIO ? 1e-6f : static_cast<float>(*eps);
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
        if (c.topology == EncoderTopology::OCR || c.topology == EncoderTopology::OCR2) {
            c.patch = 16;
            c.merge = 4;
            if (c.topology == EncoderTopology::OCR) {
                c.eps = 1e-5f;
                c.activation = "GGML_UNARY_OP_GELU_QUICK";
            } else {
                c.activation = "GGML_UNARY_OP_SILU";
            }
            return c;
        }
        if (c.topology == EncoderTopology::PIXTRAL || c.topology == EncoderTopology::GEMMA4 ||
            c.topology == EncoderTopology::UNIFIED_VISION || c.topology == EncoderTopology::MINICPM46 ||
            c.topology == EncoderTopology::PHI4) {
            if (c.topology == EncoderTopology::PIXTRAL)
                c.merge = meta.get_int(key + "spatial_merge_size").value_or(1);
            else if (c.topology != EncoderTopology::PHI4)
                c.merge = meta.get_int(key + "projector.scale_factor")
                              .value_or(c.topology == EncoderTopology::MINICPM46 ? 4 : 3);
            OPENVINO_ASSERT(c.merge > 0, "[GGUF] invalid vision merge size");
            if (c.topology == EncoderTopology::UNIFIED_VISION) {
                c.patch *= c.merge;
                c.merge = 1;
            }
            if (c.topology == EncoderTopology::MINICPM46) {
                auto layers = meta.get_int_array(key + "wa_layer_indexes");
                c.window_pattern = layers.empty() ? 0 : layers.front();
                OPENVINO_ASSERT(c.merge == 4 && c.window_pattern >= 0 && c.window_pattern < c.layers,
                                "[GGUF] invalid minicpmv4_6 merger configuration");
            }
            return c;
        }
        if (c.topology == EncoderTopology::MUSE_GLIMMER) {
            c.merge = meta.get_int(key + "spatial_merge_size").value_or(2);
            c.window_pattern = 4;
            c.activation = "GGML_UNARY_OP_GELU_ERF";
            OPENVINO_ASSERT(c.merge > 0 && c.width / c.heads % 4 == 0,
                            "[GGUF] invalid Muse Glimmer merge or rotary head dimensions");
            return c;
        }
        if (c.topology == EncoderTopology::RESAMPLER) {
            c.version = meta.get_int("clip.minicpmv_version").value_or(2);
            if (c.version == 0)
                c.version = 2;
            OPENVINO_ASSERT(c.version == 2 || c.version == 3 || c.version == 4 || c.version == 5 || c.version == 6 ||
                                c.version == 100045,
                            "[GGUF] unsupported vision mmproj projector 'resampler' version ",
                            c.version);
            c.queries = meta.get_int("clip.minicpmv_query_num").value_or(0);
            OPENVINO_ASSERT(c.queries >= 0, "[GGUF] vision resampler query count must be nonnegative");
            if (c.queries == 0)
                c.queries = c.version == 2 ? 96 : 64;
            return c;
        }
        if (c.topology == EncoderTopology::QWEN) {
            c.merge = meta.get_int(key + "spatial_merge_size").value_or(2);
            c.window_pattern = c.projector == "qwen2.5vl_merger" ? positive(meta, key + "n_wa_pattern") : 0;
            OPENVINO_ASSERT(c.merge == 2 && c.width / c.heads % 4 == 0,
                            "[GGUF] Qwen vision requires merge size two and head width divisible by four");
            return c;
        }
        c.image_size = positive(meta, key + "image_size");
        c.merge = c.projector == "janus_pro" || c.topology == EncoderTopology::CLIP
                      ? 1
                      : meta.get_int(key + "projector.scale_factor").value_or(c.projector == "gemma3" ? 4 : 2);
        OPENVINO_ASSERT(c.merge > 0 && c.image_size % c.patch == 0 && (c.image_size / c.patch) % c.merge == 0,
                        "[GGUF] incompatible mmproj patch/merge dimensions");
    }
    return c;
}

class BuiltinProjectorBuilder {
public:
    explicit BuiltinProjectorBuilder(GgufGraphContext& graph) : g(graph) {}

    ProjectorResult build(const std::string& modality, const std::string& projector) {
        auto c = config(g.metadata(), modality, projector);
        c.rms = rms_encoder(c);
        clippable = c.topology == EncoderTopology::GEMMA4 || c.topology == EncoderTopology::GEMMA4_AUDIO;
        ProjectorResult result;
        result.output = modality == "vision" ? vision(c) : audio(c);
        result.config[modality + ".merge"] = std::to_string(c.merge);
        if (c.topology == EncoderTopology::MUSE_GLIMMER)
            result.config["vision.window_size"] = std::to_string(muse_window);
        if (c.topology == EncoderTopology::RESAMPLER) {
            result.config["vision.minicpmv_version"] = std::to_string(c.version);
            result.config["vision.query_count"] = std::to_string(c.queries);
        }
        if (modality == "vision") {
            result.config["vision.auxiliary_count"] = std::to_string(auxiliary.size());
            result.config["vision.patch_size"] = std::to_string(c.patch);
        }
        return result;
    }

private:
    GgufGraphContext& g;
    std::vector<GgufValue> auxiliary;
    GgufValue default_clip_min, default_clip_max;
    // Set once per encoder in build(); the Gemma4 families clamp every linear's input and output.
    bool clippable = false;
    // Muse Glimmer window side, which matches its square learned position grid.
    int64_t muse_window = 0;

    GgufValue reshape(const GgufValue& x, std::vector<int64_t> shape, bool special_zero = false) {
        return g.node("GGML_OP_RESHAPE",
                      {x},
                      0,
                      {{"reshape_target", std::move(shape)}, {"special_zero", special_zero}});
    }
    GgufValue transpose(const GgufValue& x, std::vector<int64_t> perm = {0, 1, 3, 2}) {
        return g.node("GGML_OP_TRANSPOSE", {x}, 0, {{"perm", std::move(perm)}});
    }
    GgufValue add(const GgufValue& x, const GgufValue& y) {
        return g.node("GGML_OP_ADD", {x, y});
    }
    GgufValue mul(const GgufValue& x, const GgufValue& y) {
        return g.node("GGML_OP_MUL", {x, y});
    }
    GgufValue scale(const GgufValue& x, float value, float bias = 0.f) {
        return g.node("GGML_OP_SCALE", {x}, 0, {{"scale", value}, {"bias", bias}});
    }
    GgufValue slice(const GgufValue& x, int64_t axis, int64_t begin, int64_t count) {
        return g.node("GGML_OP_VIEW", {x}, 3, {{"view_slice", std::vector<int64_t>{axis, begin, count}}});
    }
    GgufValue concat(const GgufValue& a, const GgufValue& b, int64_t axis = 0) {
        return g.node("GGML_OP_CONCAT", {a, b}, 0, {{"concat_axis", axis}});
    }
    GgufValue grid(const GgufValue& x, const GgufValue& reference, int64_t width) {
        return reshape_like(x, reference, {1, width, 0, 0}, {-1, -1, 2, 3});
    }
    GgufValue clip_linear(const GgufValue& x, const std::string& base, const std::string& side) {
        auto lo = g.tensors()(base + "." + side + "_min"), hi = g.tensors()(base + "." + side + "_max");
        if (!lo && !hi)
            return x;
        const auto bound = [&](GgufValue& slot, const char* name, float value) {
            if (!slot) {
                ov::Tensor t(ov::element::f32, {1});
                t.data<float>()[0] = value;
                slot = g.add_constant(name, t);
            }
            return slot;
        };
        const auto limit = std::numeric_limits<float>::max();
        return g.node("GGML_OP_CLAMP",
                      {x,
                       lo ? lo : bound(default_clip_min, "mmproj.clip_min", -limit),
                       hi ? hi : bound(default_clip_max, "mmproj.clip_max", limit)});
    }
    GgufValue linear(const GgufValue& x, const std::string& base, bool with_bias = true) {
        auto y = g.node("GGML_OP_MUL_MAT",
                        {g.tensors().require(base + ".weight"), clippable ? clip_linear(x, base, "input") : x});
        if (clippable)
            y = clip_linear(y, base, "output");
        if (auto bias = with_bias ? g.tensors()(base + ".bias") : GgufValue{})
            y = add(y, bias);
        return y;
    }
    GgufValue norm(const GgufValue& x, const std::string& base, float eps) {
        return g.build_norm_ln(x, g.tensors()(base + ".weight"), g.tensors()(base + ".bias"), eps);
    }
    GgufValue encoder_norm(const GgufValue& x, const std::string& base, const EncoderConfig& c, bool with_bias = true) {
        const auto weight = g.tensors()(base + ".weight");
        const auto bias = with_bias ? g.tensors()(base + ".bias") : GgufValue{};
        if (!c.rms)
            return g.build_norm_ln(x, weight, bias, c.eps);
        auto y = g.build_norm(x, weight, c.eps);
        return bias ? add(y, bias) : y;
    }
    GgufValue attention(const GgufValue& q,
                        const GgufValue& k,
                        const GgufValue& v,
                        float factor,
                        const GgufValue& mask = {}) {
        std::vector<GgufValue> inputs{q, k, v};
        if (mask)
            inputs.push_back(mask);
        return g.node("GGML_OP_FLASH_ATTN_EXT", inputs, 100, {{"f32_attention", true}, {"scale", factor}});
    }
    GgufValue patch_embeddings(const GgufValue& spatial, int64_t width, bool with_bias = true) {
        auto x = transpose(reshape(spatial, {1, 1, width, -1}));
        if (auto bias = with_bias ? g.tensors()("v.patch_embd.bias") : GgufValue{})
            x = add(x, bias);
        return x;
    }
    GgufValue ffn(const GgufValue& x,
                  const std::string& up,
                  const std::string& down,
                  const std::string& activation,
                  const std::string& gate = "",
                  bool with_bias = true) {
        auto y = linear(x, up, with_bias);
        if (!gate.empty() && g.tensors().has(gate + ".weight"))
            y = mul(y, g.node(activation, {linear(x, gate)}));
        else
            y = g.node(activation, {y});
        return linear(y, down, with_bias);
    }
    GgufValue pool(const GgufValue& x, int64_t kx, int64_t ky) {
        // op_case 2: average pooling; params are kx, ky, sx, sy, px, py.
        return g.node(
            "GGML_OP_POOL_2D",
            {x},
            2,
            {{"pool_params", std::vector<int32_t>{int32_t(kx), int32_t(ky), int32_t(kx), int32_t(ky), 0, 0}}});
    }
    GgufValue vit(GgufValue x,
                  const EncoderConfig& c,
                  const GgufValue& positions,
                  const GgufValue& rope_positions = {},
                  const GgufValue& window_mask = {},
                  const GgufValue& rope_positions_b = {},
                  int64_t first = 0,
                  int64_t end = -1,
                  bool post_norm = true) {
        if (end < 0)
            end = c.layers;
        if (positions)
            x = add(x, positions);
        if (first == 0 && g.tensors().has(c.prefix + "pre_ln.weight"))
            x = encoder_norm(x, c.prefix + "pre_ln", c);
        // The per-layer width check below confirms a legacy ffn_up/ffn_down name swap.
        const bool swappable = c.traits & SWAPPED_FFN;
        std::vector<GgufValue> features;
        const auto save_feature = [&](int64_t layer, const GgufValue& value) {
            if (std::find(c.feature_layers.begin(), c.feature_layers.end(), layer) != c.feature_layers.end())
                features.push_back(value);
        };
        for (int64_t i = first; i < end; ++i) {
            save_feature(i, x);
            const auto p = c.prefix + "blk." + std::to_string(i) + ".";
            auto z = encoder_norm(x, p + "ln1", c);
            const bool fused = g.tensors().has(p + "attn_qkv.weight");
            auto qkv = fused ? linear(z, p + "attn_qkv") : GgufValue{};
            // Whether QK norm weights are per head or whole-tensor is a property of the layer.
            const bool per_head = fused || (g.tensors().has(p + "attn_q_norm.weight") &&
                                            g.tensors().require(p + "attn_q_norm.weight").ne(0) == c.width / c.heads);
            const auto projection = [&](const std::string& name, int64_t offset) {
                auto value = fused ? slice(qkv, 3, offset * c.width, c.width) : linear(z, p + name);
                const auto weight = g.tensors()(p + name + "_norm.weight");
                if (weight && !per_head)
                    value = encoder_norm(value, p + name + "_norm", c, false);
                value = reshape(value, {0, -1, name == "attn_q" ? c.heads : c.kv_heads, c.width / c.heads}, true);
                if (weight && per_head)
                    value = encoder_norm(value, p + name + "_norm", c, false);
                return value;
            };
            auto q = projection("attn_q", 0);
            auto k = projection("attn_k", 1);
            auto v = projection("attn_v", 2);
            if (rope_positions) {
                q = encoder_rope(q, c, rope_positions, rope_positions_b);
                k = encoder_rope(k, c, rope_positions, rope_positions_b);
            }
            if (c.topology == EncoderTopology::GEMMA4)
                v = g.build_norm(v, {}, c.eps);
            const bool muse = c.topology == EncoderTopology::MUSE_GLIMMER;
            const bool windowed = window_mask && (c.window_pattern == 0 || (i + 1) % c.window_pattern != 0) &&
                                  !(muse && i == c.layers - 1);
            auto mask = windowed ? window_mask : GgufValue{};
            // Muse Glimmer windows are padded to equal size and attend as a batch; its global
            // layer still ignores the padding.
            const int64_t window_tokens = muse && windowed ? muse_window * muse_window : 0;
            if (window_tokens) {
                const auto head = c.width / c.heads;
                q = reshape(q, {-1, window_tokens, c.heads, head});
                k = reshape(k, {-1, window_tokens, c.kv_heads, head});
                v = reshape(v, {-1, window_tokens, c.kv_heads, head});
            } else if (muse && window_mask) {
                mask = reshape(window_mask, {1, 1, 1, -1});
            }
            z = attention(q,
                          k,
                          v,
                          c.topology == EncoderTopology::GEMMA4 ? 1.f : 1.f / std::sqrt(float(c.width / c.heads)),
                          mask);
            z = window_tokens ? reshape(z, {1, 1, -1, c.width}) : reshape(z, {0, 1, -1, c.width}, true);
            z = linear(z, p + "attn_out");
            if (auto scale = g.tensors()(p + "ls1.weight"))
                z = mul(z, scale);
            if (g.tensors().has(p + "attn_post_norm.weight"))
                z = encoder_norm(z, p + "attn_post_norm", c, false);
            x = add(x, z);
            const bool legacy_swap = swappable && g.tensors().require(p + "ffn_down.weight").ne(0) == c.width;
            z = ffn(encoder_norm(x, p + "ln2", c),
                    p + (legacy_swap ? "ffn_down" : "ffn_up"),
                    p + (legacy_swap ? "ffn_up" : "ffn_down"),
                    c.activation,
                    p + "ffn_gate");
            if (g.tensors().has(p + "ffn_post_norm.weight"))
                z = encoder_norm(z, p + "ffn_post_norm", c, false);
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
        if (c.traits & POOL_2)
            x = transpose(pool(transpose(x), 2, 1));
        if (post_norm && end == c.layers && g.tensors().has(c.prefix + "post_ln.weight"))
            x = encoder_norm(x, c.prefix + "post_ln", c);
        save_feature(end, x);
        if (!features.empty()) {
            x = features.front();
            for (size_t i = 1; i < features.size(); ++i)
                x = concat(x, features[i]);
        }
        return x;
    }
    GgufValue encoder_rope(const GgufValue& x,
                           const EncoderConfig& c,
                           const GgufValue& positions,
                           const GgufValue& positions_b) {
        const auto head = int(c.width / c.heads);
        RopeConfig r;
        r.freq_scale = r.attn_factor = 1.f;
        const auto rope = [&](const GgufValue& value, const GgufValue& pos, int mode) {
            return g.node("GGML_OP_ROPE", {value, pos}, mode, {{"rope_config", r}});
        };
        if (c.topology == EncoderTopology::OCR2) {
            r.n_dims = head;
            r.freq_base = 1000000.f;
            return rope(x, positions, ROPE_NEOX);
        }
        r.n_dims = head / 2;
        if (!positions_b) {
            r.freq_base = 10000.f;
            r.sections.fill(int32_t(head / 4));
            return rope(x, positions, ROPE_VISION);
        }
        // Two position axes, each rotating one half of the head.
        const bool gemma4 = c.topology == EncoderTopology::GEMMA4;
        const int mode = gemma4 ? ROPE_NEOX : 0;
        r.freq_base = gemma4 ? 100.f : 10000.f;
        auto a = rope(slice(x, 3, 0, r.n_dims), positions, mode);
        if (c.topology == EncoderTopology::PIXTRAL)
            r.freq_scale = std::pow(r.freq_base, -2.f / float(head));
        auto b = rope(slice(x, 3, r.n_dims, r.n_dims), positions_b, mode);
        return concat(a, b);
    }
    // Resizes a square [side*side, width] learned position table to the grid of `like`; NCHW result.
    GgufValue resize_square_table(const GgufValue& table,
                                  const GgufValue& like,
                                  int64_t width,
                                  int interpolation_mode,
                                  const char* family) {
        const int64_t side = int64_t(std::sqrt(double(table.ne(1))));
        OPENVINO_ASSERT(side * side == table.ne(1), "[GGUF] ", family, " position table must be square");
        return g.node("GGML_OP_UPSCALE",
                      {transpose(reshape(table, {1, side, side, width}), {0, 3, 1, 2}), like},
                      0,
                      {{"resize_like", true}, {"interpolation_mode", interpolation_mode}});
    }
    GgufValue index_input(const std::string& name) {
        return g.add_input("vision." + name, ov::element::i32, {1, 1, 1, -1});
    }
    GgufValue gather_rows(const GgufValue& x, const std::string& index_name) {
        return g.node("GGML_OP_GET_ROWS", {x, index_input(index_name)});
    }
    GgufValue vision(const EncoderConfig& c) {
        if (c.topology == EncoderTopology::OCR || c.topology == EncoderTopology::OCR2)
            return ocr_vision(c);
        if (c.topology == EncoderTopology::PIXTRAL || c.topology == EncoderTopology::GEMMA4 ||
            c.topology == EncoderTopology::UNIFIED_VISION || c.topology == EncoderTopology::PHI4)
            return dynamic_vision(c);
        if (c.topology == EncoderTopology::MINICPM46)
            return minicpm46(c);
        if (c.topology == EncoderTopology::QWEN)
            return qwen_vision(c);
        if (c.topology == EncoderTopology::MUSE_GLIMMER)
            return muse_glimmer_vision(c);
        if (c.topology == EncoderTopology::RESAMPLER)
            return resampler_vision(c);
        auto x = g.add_input("vision.pixel_values", ov::element::f32, {1, 3, c.image_size, c.image_size});
        x = patch_embeddings(convolution(x, "v.patch_embd.weight", c.patch), c.width);
        if ((c.topology == EncoderTopology::CLIP || c.topology == EncoderTopology::INTERNVL) &&
            g.tensors().has("v.class_embd"))
            x = concat(x, reshape(g.tensors().require("v.class_embd"), {1, 1, 1, c.width}), 1);
        x = vit(x, c, g.tensors().require("v.position_embd.weight"));
        const auto side = c.image_size / c.patch;
        if (c.topology == EncoderTopology::INTERNVL) {
            OPENVINO_ASSERT(g.tensors().has("v.class_embd"), "[GGUF] InternVL requires a class embedding");
            x = slice(x, 2, 0, side * side);
            x = reshape(x, {1, side, side / c.merge, c.width * c.merge});
            x = transpose(x, {0, 2, 1, 3});
            x = reshape(x, {1, side / c.merge, side / c.merge, c.width * c.merge * c.merge});
            x = transpose(x, {0, 2, 1, 3});
            x = reshape(x, {1, 1, -1, c.width * c.merge * c.merge});
            return ffn(norm(x, "mm.model.mlp.0", 1e-5f), "mm.model.mlp.1", "mm.model.mlp.3", "GGML_UNARY_OP_GELU");
        }
        if (c.topology == EncoderTopology::CLIP) {
            const int64_t offset = g.tensors().has("v.class_embd") ? 1 : 0;
            x = slice(x, 2, offset, side * side);
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
    GgufValue convolution(const GgufValue& pixels, const GgufValue& weight, int64_t stride, int64_t padding = 0) {
        return g.node("GGML_OP_CONV_2D",
                      {weight, pixels},
                      0,
                      {{"conv_params", std::vector<int64_t>{stride, stride, padding, padding, 1, 1}}});
    }
    GgufValue convolution(const GgufValue& pixels, const std::string& weight, int64_t stride, int64_t padding = 0) {
        return convolution(pixels, g.tensors().require(weight), stride, padding);
    }
    GgufValue unfold(const GgufValue& x, int64_t channels, int64_t kernel) {
        return g.node("GGML_OP_IM2COL",
                      {x},
                      0,
                      {{"im2col_params", std::vector<int32_t>{int32_t(kernel), int32_t(kernel), 0, 0, 1, 1, 1}},
                       {"kernel_shape", ov::Shape{1, size_t(channels), size_t(kernel), size_t(kernel)}},
                       {"dst_type", ov::element::f32}});
    }
    GgufValue dynamic_vision(const EncoderConfig& c) {
        auto pixels = g.add_input("vision.pixel_values", ov::element::f32, {1, 3, -1, -1});
        GgufValue spatial, x;
        if (c.topology == EncoderTopology::UNIFIED_VISION) {
            for (int i = 1; i <= 3; ++i) {
                g.tensors().require("v.patch_norm." + std::to_string(i) + ".weight");
                g.tensors().require("v.patch_norm." + std::to_string(i) + ".bias");
            }
            spatial = pool(pixels, c.patch, c.patch);
            x = reshape(unfold(pixels, 3, c.patch), {1, 1, -1, 3 * c.patch * c.patch});
            // LayerNorm is invariant to a uniform shift. Recenter low-contrast pixel patches
            // before the mean reduction, whose F32 accumulation otherwise loses precision.
            x = g.node("GGML_OP_SUB", {x, slice(x, 3, 0, 1)});
            x = norm(x, "v.patch_norm.1", 1e-5f);
            // The patch projection reaches 1.5e5 on real images, beyond F16. LayerNorm is invariant
            // to a uniform scale once its eps shrinks by the square, so compute it shrunk.
            x = linear(scale(x, kUnifiedShrink), "v.patch_embd", /*with_bias=*/false);
            x = add(x, scale(g.tensors().require("v.patch_embd.bias"), kUnifiedShrink));
            x = norm(x, "v.patch_norm.2", 1e-5f * kUnifiedShrink * kUnifiedShrink);
        } else {
            spatial = convolution(c.topology == EncoderTopology::GEMMA4 ? scale(pixels, 2.f, -1.f) : pixels,
                                  "v.patch_embd.weight",
                                  c.patch);
            x = patch_embeddings(spatial, c.width, c.topology != EncoderTopology::GEMMA4);
        }
        GgufValue pos_a, pos_b, learned;
        if (c.topology == EncoderTopology::PHI4) {
            auto table =
                resize_square_table(g.tensors().require("v.position_embd.weight"), spatial, c.width, 0x201, "phi4");
            learned = reshape(transpose(table, {0, 2, 3, 1}), {1, 1, -1, c.width});
        } else {
            pos_a = index_input("position_x");
            pos_b = index_input("position_y");
            if (c.topology != EncoderTopology::PIXTRAL) {
                auto table = g.tensors().require("v.position_embd.weight");
                OPENVINO_ASSERT(table.ne(2) == 2, "[GGUF] Gemma4 position table requires x/y axes");
                table = reshape(table, {1, 2, table.ne(1), c.width});
                learned = add(g.node("GGML_OP_GET_ROWS", {slice(table, 1, 0, 1), pos_a}),
                              g.node("GGML_OP_GET_ROWS", {slice(table, 1, 1, 1), pos_b}));
            }
        }
        if (c.topology == EncoderTopology::UNIFIED_VISION) {
            // The RMSNorm input squared exceeds F16 (x up to ~800); RMSNorm is scale-invariant as well.
            auto normed = scale(norm(add(x, learned), "v.patch_norm.3", 1e-5f), kUnifiedShrink);
            return linear(g.build_norm(normed, {}, c.eps * kUnifiedShrink * kUnifiedShrink), "mm.input_projection");
        }
        // Pixtral uses height then width, with alternating frequencies; Gemma4 uses x then y, NEOX halves.
        x = vit(x,
                c,
                learned,
                c.topology == EncoderTopology::PIXTRAL ? pos_b : pos_a,
                {},
                c.topology == EncoderTopology::PIXTRAL ? pos_a : pos_b);
        if (c.topology == EncoderTopology::PHI4) {
            g.tensors().require("mm.0.bias");
            g.tensors().require("mm.2.bias");
            return ffn(x, "mm.0", "mm.2", "GGML_UNARY_OP_GELU");
        }
        if (c.topology == EncoderTopology::GEMMA4) {
            x = pool(grid(transpose(x), spatial, c.width), c.merge, c.merge);
            x = scale(transpose(reshape(x, {1, 1, c.width, -1})), std::sqrt(float(c.width)));
            if (g.tensors().has("v.std_bias") && g.tensors().has("v.std_scale"))
                x = mul(g.node("GGML_OP_SUB", {x, g.tensors().require("v.std_bias")}),
                        g.tensors().require("v.std_scale"));
            return linear(g.build_norm(x, {}, c.eps), "mm.input_projection");
        }
        if (g.tensors().has("mm.patch_merger.weight")) {
            x = g.build_norm(x, g.tensors().require("mm.input_norm.weight"), c.eps);
            x = unfold(grid(transpose(x), spatial, c.width), c.width, c.merge);
            x = linear(reshape(x, {1, 1, -1, c.width * c.merge * c.merge}), "mm.patch_merger");
            spatial = pool(spatial, c.merge, c.merge);
        } else {
            OPENVINO_ASSERT(c.merge == 1, "[GGUF] pixtral merge requires patch merger weights");
        }
        x = ffn(x, "mm.1", "mm.2", "GGML_UNARY_OP_GELU");
        if (auto token = g.tensors()("v.token_embd.img_break")) {
            const auto width = token.ne(0);
            x = transpose(grid(transpose(x), spatial, width), {0, 2, 3, 1});
            auto row_break = add(scale(slice(x, 2, 0, 1), 0.f), token);
            x = reshape(concat(x, row_break, 1), {1, 1, -1, width});
            x = slice(x, 2, 0, -1);  // no break after the final row
        }
        return x;
    }
    GgufValue minicpm46(const EncoderConfig& c) {
        auto pixels = g.add_input("vision.pixel_values", ov::element::f32, {1, 3, -1, -1});
        auto x = patch_embeddings(convolution(pixels, "v.patch_embd.weight", c.patch), c.width);
        auto learned = gather_rows(g.tensors().require("v.position_embd.weight"), "position_ids");
        x = vit(x, c, learned, {}, {}, {}, 0, c.window_pattern + 1, false);
        const std::string p = "v.vit_merger.";
        auto z = gather_rows(norm(x, p + "ln1", c.eps), "window_indices");
        auto q = reshape(linear(z, p + "attn_q"), {1, -1, c.heads, c.width / c.heads});
        auto k = reshape(linear(z, p + "attn_k"), {1, -1, c.heads, c.width / c.heads});
        auto v = reshape(linear(z, p + "attn_v"), {1, -1, c.heads, c.width / c.heads});
        auto mask = g.add_input("vision.attention_mask", ov::element::f32, {1, 1, -1, -1});
        z = attention(q, k, v, 1.f / std::sqrt(float(c.width / c.heads)), mask);
        z = linear(reshape(z, {1, 1, -1, c.width}), p + "attn_out");
        x = add(x, gather_rows(z, "inverse_window_indices"));
        // Gather the four cells of each 2x2 merge window, in index order.
        const auto gather4 = [&](const GgufValue& value, const std::string& name) {
            std::array<GgufValue, 4> parts;
            for (int i = 0; i < 4; ++i)
                parts[i] = gather_rows(value, name + ".indices." + std::to_string(i));
            return parts;
        };
        const auto concat4 = [&](const std::array<GgufValue, 4>& parts) {
            return concat(concat(concat(parts[0], parts[1]), parts[2]), parts[3]);
        };
        auto parts = gather4(x, "vit_merger");
        auto cat = concat4(parts);
        x = add(ffn(norm(cat, p + "ds_ln", c.eps), p + "ds_ffn_up", p + "ds_ffn_down", "GGML_UNARY_OP_GELU"),
                scale(add(add(add(parts[0], parts[1]), parts[2]), parts[3]), .25f));
        x = vit(x, c, {}, {}, {}, {}, c.window_pattern + 1);
        parts = gather4(x, "merger");
        cat = concat4(parts);
        return ffn(norm(cat, "mm.input_norm", c.eps), "mm.up", "mm.down", "GGML_UNARY_OP_GELU_ERF");
    }
    GgufValue resampler_vision(const EncoderConfig& c) {
        auto pixels = g.add_input("vision.pixel_values", ov::element::f32, {1, 3, -1, -1});
        auto x = patch_embeddings(convolution(pixels, "v.patch_embd.weight", c.patch), c.width);
        x = vit(x, c, gather_rows(g.tensors().require("v.position_embd.weight"), "position_ids"));

        const auto query = g.tensors().require("resampler.query");
        const auto width = query.ne(0);
        OPENVINO_ASSERT(width > 0 && width % 128 == 0 && query.ne(1) == c.queries,
                        "[GGUF] vision resampler requires 128-wide heads and a query tensor matching query_count");
        for (const auto* name : {"q", "kv", "post"}) {
            g.tensors().require(std::string("resampler.ln_") + name + ".weight");
            g.tensors().require(std::string("resampler.ln_") + name + ".bias");
        }
        for (const auto* name : {"q", "k", "v", "out"})
            g.tensors().require(std::string("resampler.attn.") + name + ".bias");
        auto q = norm(reshape(query, {1, 1, c.queries, width}), "resampler.ln_q", c.eps);
        auto v = norm(linear(x, "resampler.kv"), "resampler.ln_kv", c.eps);
        auto pos_h = g.add_input("vision.position_h", ov::element::f32, {1, 1, -1, 1});
        auto pos_w = g.add_input("vision.position_w", ov::element::f32, {1, 1, -1, 1});
        ov::Tensor omega(ov::element::f32, {1, 1, 1, size_t(width / 4)});
        for (int64_t i = 0; i < width / 4; ++i)
            omega.data<float>()[i] = 1.f / std::pow(10000.f, float(i) / float(width / 4));
        auto frequency = g.add_constant("vision.resampler_omega", omega);
        const auto sinusoid = [&](const GgufValue& pos) {
            auto theta = mul(pos, frequency);
            return concat(g.node("GGML_OP_SIN", {theta}), g.node("GGML_OP_COS", {theta}));
        };
        auto positional = concat(sinusoid(pos_w), sinusoid(pos_h));
        auto k = add(v, positional);
        q = reshape(linear(q, "resampler.attn.q"), {1, c.queries, width / 128, 128});
        k = reshape(linear(k, "resampler.attn.k"), {1, -1, width / 128, 128});
        v = reshape(linear(v, "resampler.attn.v"), {1, -1, width / 128, 128});
        x = attention(q, k, v, 1.f / std::sqrt(128.f));
        x = linear(reshape(x, {1, 1, c.queries, width}), "resampler.attn.out");
        return linear(norm(x, "resampler.ln_post", c.eps), "resampler.proj");
    }
    GgufValue muse_glimmer_vision(const EncoderConfig& c) {
        auto pixels = g.add_input("vision.pixel_values", ov::element::f32, {1, 3, -1, -1});
        auto spatial = convolution(pixels, "v.patch_embd.weight", c.patch);
        auto x = patch_embeddings(spatial, c.width);
        auto table = g.tensors().require("v.position_embd.weight");
        table = resize_square_table(table, spatial, c.width, 1, "Muse Glimmer");
        x = add(x, reshape(transpose(table, {0, 2, 3, 1}), {1, 1, -1, c.width}));
        x = gather_rows(x, "patch_indices");
        auto pos_x = index_input("position_x"), pos_y = index_input("position_y");
        muse_window = int64_t(std::sqrt(double(g.tensors().require("v.position_embd.weight").ne(1))));
        // Additive key mask of every window padded to muse_window^2 patches.
        auto mask = g.add_input("vision.window_mask", ov::element::f32, {-1, 1, 1, muse_window * muse_window});
        x = vit(x, c, {}, pos_x, mask, pos_y);
        x = gather_rows(x, "output_indices");
        x = gather_rows(x, "merge_indices");
        // Channel-outer pixel shuffle: each channel's spatial neighbours stay together.
        x = reshape(x, {1, -1, c.merge * c.merge, c.width});
        x = reshape(transpose(x, {0, 1, 3, 2}), {1, 1, -1, c.width * c.merge * c.merge});
        x = g.node("GGML_UNARY_OP_GELU_ERF", {linear(x, "mm.0", false)});
        x = g.node("GGML_UNARY_OP_GELU_ERF", {linear(x, "mm.1", false)});
        return linear(x, "mm.2", false);
    }
    GgufValue qwen_vision(const EncoderConfig& c) {
        // A temporal pair; still-image callers duplicate the image. Index inputs specify
        // spatial 2x2 grouping and, for Qwen2.5, the reference's window permutation.
        auto pixels = g.add_input("vision.pixel_values", ov::element::f32, {2, 3, -1, -1});
        // One convolution over both frames stacked on the channel axis.
        auto weight =
            concat(g.tensors().require("v.patch_embd.weight"), g.tensors().require("v.patch_embd.weight.1"), 2);
        auto patches = convolution(reshape(pixels, {1, -1, 0, 0}, true), weight, c.patch);
        auto indices = index_input("patch_indices");
        auto group = [&](const GgufValue& value) {
            return g.node("GGML_OP_GET_ROWS", {transpose(reshape(value, {1, 1, c.width, -1})), indices});
        };
        auto x = group(patches);
        if (auto bias = g.tensors()("v.patch_embd.bias"))
            x = add(x, bias);
        GgufValue learned_positions;
        if (c.projector == "qwen3vl_merger") {
            learned_positions = group(resize_square_table(g.tensors().require("v.position_embd.weight"),
                                                          patches,
                                                          c.width,
                                                          1 | 0x100,
                                                          "Qwen"));
        }
        auto positions = index_input("position_ids");
        GgufValue window_mask;
        if (c.window_pattern)
            window_mask = g.add_input("vision.attention_mask", ov::element::f32, {1, 1, -1, -1});
        x = vit(x, c, learned_positions, positions, window_mask);
        x = ffn(reshape(x, {1, 1, -1, 4 * c.width}), "mm.0", "mm.2", "GGML_UNARY_OP_GELU");
        for (const auto& feature : auxiliary)
            x = concat(x, feature);
        if (c.window_pattern)
            x = gather_rows(x, "output_indices");
        return x;
    }
    GgufValue reshape_like(const GgufValue& x,
                           const GgufValue& reference,
                           std::vector<int64_t> pattern,
                           std::vector<int64_t> axes) {
        return g.node("GGML_OP_RESHAPE",
                      {x, reference},
                      0,
                      {{"reshape_target", std::move(pattern)}, {"shape_axes", std::move(axes)}});
    }
    GgufValue sam(const GgufValue& pixels) {
        const auto width = positive(g.metadata(), "clip.vision.sam.embedding_length");
        const auto heads = positive(g.metadata(), "clip.vision.sam.head_count");
        const auto layers = positive(g.metadata(), "clip.vision.sam.block_count");
        const auto window = positive(g.metadata(), "clip.vision.window_size");
        OPENVINO_ASSERT(width % heads == 0, "[GGUF] invalid SAM head count");
        const auto head = width / heads;
        auto x = convolution(pixels, "v.sam.patch_embd.weight", g.tensors().require("v.sam.patch_embd.weight").ne(0));
        x = transpose(x, {0, 2, 3, 1});
        x = add(x, g.tensors().require("v.sam.patch_embd.bias"));
        auto pos = g.tensors().require("v.sam.pos_embd.weight");
        pos = reshape(pos, {1, pos.ne(2), pos.ne(1), width});
        pos = g.node("GGML_OP_UPSCALE",
                     {transpose(pos, {0, 3, 1, 2}), transpose(x, {0, 3, 1, 2})},
                     0,
                     {{"resize_like", true}, {"interpolation_mode", 2}});
        x = add(x, transpose(pos, {0, 2, 3, 1}));
        auto local = g.add_input("vision.relative_indices_local", ov::element::i32, {1, 1, window, window});
        auto global = g.add_input("vision.relative_indices_global", ov::element::i32, {1, 1, -1, -1});
        for (int64_t i = 0; i < layers; ++i) {
            const auto p = "v.sam.blk." + std::to_string(i) + ".";
            const bool global_layer = i == 2 || i == 5 || i == 8 || i == 11;
            auto z = norm(x, p + "pre_ln", 1e-6f);
            if (!global_layer)
                z = g.node("GGML_OP_WIN_PART", {z}, 0, {{"window", window}});
            auto geometry = z;
            auto qkv = linear(z, p + "attn.qkv");
            const auto project = [&](int part) {
                return reshape(slice(qkv, 3, part * width, width), {0, -1, heads, head}, true);
            };
            auto q = project(0), k = project(1), v = project(2);
            auto qr = reshape_like(transpose(q, {0, 2, 1, 3}), geometry, {-1, 0, 0, head}, {-1, 1, 2, -1});
            const auto rel_pos = [&](const std::string& axis) {
                return g.node("GGML_OP_GET_REL_POS",
                              {g.tensors().require(p + "attn.pos_" + axis + ".weight"), global_layer ? global : local},
                              1);
            };
            auto rw = rel_pos("w");
            auto rh = rel_pos("h");
            rw = transpose(g.node("GGML_OP_MUL_MAT", {rw, transpose(qr, {0, 2, 1, 3})}), {0, 2, 1, 3});
            rh = g.node("GGML_OP_MUL_MAT", {rh, qr});
            rw = reshape_like(rw, rw, {0, 0, 0, 1, 0}, {0, 1, 2, -1, 3});
            rh = reshape_like(rh, rh, {0, 0, 0, 0, 1}, {0, 1, 2, 3, -1});
            auto mask = reshape_like(add(rw, rh), q, {0, heads, 0, 0}, {0, -1, 1, 1});
            z = attention(q, k, v, 1.f / std::sqrt(float(head)), mask);
            z = linear(reshape_like(z, geometry, {0, 0, 0, 0}, {0, 1, 2, 3}), p + "attn.out");
            if (!global_layer)
                z = g.node("GGML_OP_WIN_UNPART", {z, x}, 0, {{"window", window}});
            x = add(x, z);
            x = add(x, ffn(norm(x, p + "post_ln", 1e-6f), p + "mlp.lin1", p + "mlp.lin2", "GGML_UNARY_OP_GELU"));
        }
        x = convolution(transpose(x, {0, 3, 1, 2}), "v.sam.neck.0.weight", 1);
        x = norm(transpose(x, {0, 2, 3, 1}), "v.sam.neck.1", 1e-6f);
        x = convolution(transpose(x, {0, 3, 1, 2}), "v.sam.neck.2.weight", 1, 1);
        x = norm(transpose(x, {0, 2, 3, 1}), "v.sam.neck.3", 1e-6f);
        x = convolution(transpose(x, {0, 3, 1, 2}), "v.sam.net_2.weight", 2, 1);
        return convolution(x, "v.sam.net_3.weight", 2, 1);
    }
    GgufValue ocr_vision(const EncoderConfig& c) {
        auto pixels = g.add_input("vision.pixel_values", ov::element::f32, {-1, 3, -1, -1});
        auto spatial = sam(pixels);
        const auto sam_features = reshape(transpose(spatial, {0, 2, 3, 1}), {0, 1, -1, c.width}, true);
        auto x = sam_features;
        if (c.topology == EncoderTopology::OCR2) {
            auto queries = concat(reshape(g.tensors().require("v.resample_query_768.weight"), {1, 1, 144, c.width}),
                                  reshape(g.tensors().require("v.resample_query_1024.weight"), {1, 1, 256, c.width}),
                                  1);
            x = concat(x, gather_rows(queries, "query_indices"), 1);
            auto positions = index_input("position_ids");
            auto mask = g.add_input("vision.attention_mask", ov::element::f32, {1, 1, -1, -1});
            x = vit(x, c, {}, positions, mask);
            x = g.node("GGML_OP_GET_ROWS", {x, index_input("query_output_indices")}, 4);
        } else {
            auto cls = reshape(g.tensors().require("v.class_embd"), {1, 1, 1, c.width});
            // Repeat the class token once per independently encoded tile.
            cls = add(scale(slice(x, 2, 0, 1), 0.f), cls);
            x = concat(cls, x, 1);
            auto pos = g.tensors().require("v.position_embd.weight");
            // Preserve the pinned loader's legacy CLIP table layout, including the
            // byte offset of its trailing class position. The caller selects the
            // original table for an unchanged grid, or the resized table otherwise.
            const auto side = int64_t(std::sqrt(double(pos.ne(1) - 1)));
            auto original = reshape(pos, {1, 1, -1, c.width});
            auto old = slice(original, 2, 0, side * side);
            auto resized = g.node("GGML_OP_UPSCALE",
                                  {old, slice(spatial, 0, 0, 1)},
                                  0,
                                  {{"resize_like", true}, {"interpolation_mode", 2}});
            resized = g.node("GGML_OP_REPEAT", {resized}, 0, {{"repeats", std::vector<int64_t>{1, c.width, 1, 1}}});
            resized = reshape(resized, {1, 1, -1, c.width});
            auto cls_pos = slice(reshape(pos, {1, 1, 1, -1}), 3, side * side / int64_t(pos.type().size()), c.width);
            auto tables = concat(original, concat(resized, cls_pos, 1), 1);
            pos = gather_rows(tables, "position_indices");
            x = vit(x, c, pos);
            x = slice(x, 2, 1, std::numeric_limits<int32_t>::max() - 1);
            x = concat(x, sam_features);
        }
        x = linear(x, "mm.model.fc");
        const auto width = g.tensors().require("mm.model.fc.weight").ne(1);
        x = reshape(x, {1, 1, -1, width});
        // Indexing describes overview/tile row assembly, including learned separators.
        if (c.topology == EncoderTopology::OCR)
            x = concat(x, reshape(g.tensors().require("v.image_newline"), {1, 1, 1, width}), 1);
        x = concat(x, reshape(g.tensors().require("v.view_seperator"), {1, 1, 1, width}), 1);
        return gather_rows(x, "output_indices");
    }
    GgufValue gemma4_audio(const EncoderConfig& c) {
        const auto mel = positive(g.metadata(), "clip.audio.num_mel_bins");
        auto x = g.add_input("audio.features", ov::element::f32, {1, 1, mel, -1});
        x = transpose(x);  // [1, 1, time, frequency]
        for (int i = 0; i < 2; ++i) {
            const auto p = "a.conv1d." + std::to_string(i);
            x = convolution(x, p + ".weight", 2, 1);
            if (auto bias = g.tensors()(p + ".bias"))
                x = add(x, bias);
            x = transpose(x, {0, 2, 3, 1});
            if (auto weight = g.tensors()(p + ".norm.weight"))
                x = g.build_norm_ln(x, weight, {}, c.eps);
            x = transpose(g.node("GGML_UNARY_OP_RELU", {x}), {0, 3, 1, 2});
        }
        // Reference flatten order is frequency-major with channels innermost.
        const auto flattened = g.tensors().require("a.input_projection.weight").ne(0);
        x = reshape(transpose(x, {0, 2, 3, 1}), {1, 1, -1, flattened});
        x = linear(x, "a.input_projection");
        // Each token attends to itself and the 11 before it, with relative positions 12..0.
        constexpr int64_t horizon = 12;
        ov::Tensor table(ov::element::f32, {1, 1, horizon + 1, static_cast<size_t>(c.width)});
        const int64_t half = c.width / 2;
        const float increment = std::log(10000.f) / std::max<int64_t>(half - 1, 1);
        for (int64_t p = 0; p <= horizon; ++p)
            for (int64_t i = 0; i < half; ++i) {
                const float angle = float(horizon - p) * std::exp(-float(i) * increment);
                table.data<float>()[p * c.width + i] = std::sin(angle);
                table.data<float>()[p * c.width + half + i] = std::cos(angle);
            }
        auto positions = g.add_constant("mmproj.audio.relative_positions", table);
        const auto clamp = [&](const GgufValue& value, float lo, float hi) {
            return g.node("GGML_OP_CLAMP", {value}, 0, {{"clamp_min", lo}, {"clamp_max", hi}});
        };
        auto index = scale(g.node("GGML_OP_CUMSUM", {transpose(scale(slice(x, 3, 0, 1), 0.f, 1.f))}), 1.f, -1.f);
        auto distance = g.node("GGML_OP_SUB", {transpose(index), index});  // query - key
        const auto ahead = scale(distance, -1.f, float(horizon));
        auto relative = g.node("GGML_OP_CPY", {clamp(ahead, 0.f, float(horizon))}, 0, {{"dst_type", ov::element::i32}});
        auto inside = mul(clamp(scale(distance, 1.f, 1.f), 0.f, 1.f), clamp(ahead, 0.f, 1.f));
        auto mask = scale(inside, 1e9f, -1e9f);
        const auto rms = [&](const GgufValue& value, const std::string& name) {
            return g.build_norm(value, g.tensors().require(name + ".weight"), c.eps);
        };
        const auto maybe_rms = [&](const GgufValue& value, const std::string& name) {
            auto w = g.tensors()(name + ".weight");
            return w ? g.build_norm(value, w, c.eps) : value;
        };
        // Both FFN sublayers are post-normed and added back at half weight.
        const auto half_ffn = [&](const GgufValue& value, const std::string& p, const std::string& suffix) {
            auto z = ffn(rms(value, p + "ffn_norm" + suffix),
                         p + "ffn_up" + suffix,
                         p + "ffn_down" + suffix,
                         "GGML_UNARY_OP_SILU",
                         "",
                         false);
            return add(value, scale(maybe_rms(z, p + "ffn_post_norm" + suffix), .5f));
        };
        for (int64_t i = 0; i < c.layers; ++i) {
            const auto p = "a.blk." + std::to_string(i) + ".";
            x = half_ffn(x, p, "");
            auto z = rms(x, p + (g.tensors().has(p + "attn_pre_norm.weight") ? "attn_pre_norm" : "ln1"));
            auto q = linear(z, p + "attn_q", false);
            auto k = linear(z, p + "attn_k", false);
            auto v = linear(z, p + "attn_v", false);
            const auto head = c.width / c.heads;
            q = scale(reshape(q, {1, -1, c.heads, head}), 1.f / std::sqrt(float(head)) / std::log(2.f));
            k = scale(reshape(k, {1, -1, c.heads, head}), std::log1p(std::exp(1.f)) / std::log(2.f));
            if (auto w = g.tensors()(p + "per_dim_scale.weight"))
                q = mul(q, reshape(w, {1, 1, 1, head}));
            if (auto w = g.tensors()(p + "per_dim_k_scale.weight"))
                k = mul(k, reshape(w, {1, 1, 1, head}));
            q = transpose(q, {0, 2, 1, 3});
            k = transpose(k, {0, 2, 1, 3});
            v = transpose(reshape(v, {1, -1, c.heads, head}), {0, 2, 3, 1});
            auto scores = g.node("GGML_OP_MUL_MAT", {k, q});
            if (g.tensors().has(p + "attn_k_rel.weight")) {
                // The reference projects RPE without clamping, unlike its content projections.
                auto pos = g.node("GGML_OP_MUL_MAT", {g.tensors().require(p + "attn_k_rel.weight"), positions});
                pos = transpose(reshape(pos, {1, 13, c.heads, head}), {0, 2, 1, 3});
                auto bias = g.node("GGML_OP_MUL_MAT", {pos, q});
                bias = g.node("GGML_OP_GET_ROWS", {bias, relative}, 0, {{"gather_elements", true}});
                scores = add(scores, bias);
            }
            scores = add(scale(g.node("GGML_UNARY_OP_TANH", {scale(scores, 1.f / 50.f)}), 50.f), mask);
            auto probability = g.node("GGML_OP_SOFT_MAX", {scores}, 0, {{"scale", 1.f}});
            z = g.node("GGML_OP_MUL_MAT", {v, probability});
            z = linear(reshape(transpose(z, {0, 2, 1, 3}), {1, 1, -1, c.width}), p + "attn_out");
            x = add(x, maybe_rms(z, p + "attn_post_norm"));
            z = linear(rms(x, p + "conv_norm"), p + "conv_pw1", false);
            z = mul(slice(z, 3, 0, c.width), g.node("GGML_UNARY_OP_SIGMOID", {slice(z, 3, c.width, c.width)}));
            z = transpose(z);
            z = g.node("GGML_OP_PAD", {z}, 0, {{"pad_params", std::vector<int32_t>{4, 0, 0, 0, 0, 0, 0, 0}}});
            z = g.node("GGML_OP_SSM_CONV", {z, g.tensors().require(p + "conv_dw.weight")});
            if (auto bias = g.tensors()(p + "conv_dw.bias"))
                z = add(z, bias);
            z = maybe_rms(z, p + "norm_conv");
            x = add(x, linear(g.node("GGML_UNARY_OP_SILU", {z}), p + "conv_pw2", false));
            x = half_ffn(x, p, "_1");
            x = maybe_rms(x, p + "ln2");
        }
        if (g.tensors().has("a.pre_encode.out.weight"))
            x = linear(x, "a.pre_encode.out");
        x = g.build_norm(x, g.tensors()("mm.a.soft_emb_norm.weight"), c.eps);
        return linear(x, "mm.a.input_projection", false);
    }
    GgufValue audio(const EncoderConfig& c) {
        if (c.topology == EncoderTopology::GEMMA4_AUDIO)
            return gemma4_audio(c);
        if (c.topology == EncoderTopology::UNIFIED_AUDIO) {
            auto x = g.add_input("audio.waveform_frames", ov::element::f32, {1, 1, -1, 640});
            return linear(g.build_norm(x, {}, c.eps), "mm.a.input_projection");
        }
        const auto mel = positive(g.metadata(), "clip.audio.num_mel_bins");
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
            const auto stack = positive(g.metadata(), "clip.audio.projector.stack_factor");
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
            x = concat(reshape(g.tensors().require("v.boi"), {1, 1, 1, width}), x, 1);
            return concat(x, reshape(g.tensors().require("v.eoi"), {1, 1, 1, width}), 1);
        }
        return ffn(x, "mm.a.mlp.1", "mm.a.mlp.2", "GGML_UNARY_OP_GELU_ERF");
    }
};

class MmprojBuilder : public ModelBuilder {
public:
    explicit MmprojBuilder(const BuildContext& context) : ctx(context), g(context) {}

    std::shared_ptr<GgufGraph> build() override {
        std::optional<ProjectorRegistry> defaults;
        const auto& registry = ctx.projectors ? *ctx.projectors : defaults.emplace();
        std::vector<std::shared_ptr<const ProjectorDefinition>> encoders;
        for (const auto* modality : {"vision", "audio"}) {
            if (!ctx.metadata.get_bool(std::string("clip.has_") + modality + "_encoder").value_or(false))
                continue;
            const auto type = resolve_projector_type(ctx.metadata, modality);
            const auto definition = registry.find(ctx.metadata, modality, type);
            OPENVINO_ASSERT(definition, "[GGUF] unsupported ", modality, " mmproj projector '", type, "'");
            encoders.push_back(definition);
        }
        OPENVINO_ASSERT(!encoders.empty(), "[GGUF] mmproj has no encoder");
        std::map<std::string, std::string> branch_config;
        branch_config["vision.auxiliary_count"] = "0";
        for (const auto& definition : encoders) {
            auto result = definition->build(g);
            OPENVINO_ASSERT(result.output, "[GGUF] projector handler '", definition->id, "' returned no output");
            g.set_output(result.output, definition->modality + ".embeddings");
            for (const auto& entry : result.config) {
                OPENVINO_ASSERT(entry.first.rfind(definition->modality + ".", 0) == 0,
                                "[GGUF] projector metadata must use its modality prefix");
                branch_config[entry.first] = entry.second;
            }
            branch_config[definition->modality + ".projector"] = definition->projector_type;
        }
        auto graph = g.finish();
        const auto number = [](auto value) {
            std::ostringstream stream;
            stream.precision(std::numeric_limits<double>::max_digits10);
            stream << value;
            return stream.str();
        };
        const auto join = [&](const auto& values) {
            std::string text;
            for (size_t i = 0; i < values.size(); ++i)
                text += (i ? "," : "") + number(values[i]);
            return text;
        };
        // Preserve full metadata names. Strings serialize through the standard IR rt_info path.
        for (const auto& [key, value] : detail::MetadataAccess::get(ctx.metadata).map) {
            if (key.rfind("clip.", 0) != 0)
                continue;
            if (const auto s = ctx.metadata.get_str(key)) {
                graph->mmproj_config[key] = *s;
            } else if (const auto n = ctx.metadata.get_int(key)) {
                graph->mmproj_config[key] = std::to_string(*n);
            } else if (const auto f = ctx.metadata.get_float(key)) {
                graph->mmproj_config[key] = number(*f);
            } else if (std::holds_alternative<std::vector<std::string>>(value)) {
                // Length-prefixed strings preserve commas, quotes and empty entries.
                std::ostringstream stream;
                for (const auto& item : ctx.metadata.get_str_array(key))
                    stream << item.size() << ':' << item;
                graph->mmproj_config[key] = stream.str();
                graph->mmproj_config[key + ".encoding"] = std::string("length-prefixed-strings");
            } else if (const auto integers = ctx.metadata.get_int_array(key); !integers.empty()) {
                graph->mmproj_config[key] = join(integers);
            } else {
                graph->mmproj_config[key] = join(ctx.metadata.get_float_array(key));
            }
        }
        for (const auto& entry : branch_config)
            graph->mmproj_config[entry.first] = entry.second;
        return graph;
    }

private:
    BuildContext ctx;
    GgufGraphContext g;
};

}  // namespace

ProjectorResult build_builtin_projector(GgufGraphContext& graph,
                                        const std::string& modality,
                                        const std::string& projector_type) {
    return BuiltinProjectorBuilder(graph).build(modality, projector_type);
}

std::vector<ProjectorDefinition> builtin_projectors() {
    std::vector<ProjectorDefinition> definitions;
    for (const auto& entry : projector_catalog) {
        const std::string modality = entry.modality;
        const std::string type = entry.name;
        definitions.push_back({"clip." + modality + "." + type,
                               "clip",
                               modality,
                               type,
                               [modality, type](GgufGraphContext& graph) {
                                   return build_builtin_projector(graph, modality, type);
                               },
                               {}});
    }
    return definitions;
}

std::shared_ptr<ModelBuilder> make_mmproj_builder(const BuildContext& context) {
    return std::make_shared<MmprojBuilder>(context);
}

}  // namespace ov::frontend::gguf
