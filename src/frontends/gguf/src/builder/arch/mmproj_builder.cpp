// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mmproj_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>

#include "builder/api/metadata_store.hpp"
#include "builder/gguf_graph.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov::frontend::gguf {
namespace {

enum class EncoderTopology {
    Siglip,
    Clip,
    Whisper,
    Qwen,
    Internvl,
    Resampler,
    Pixtral,
    Gemma4,
    UnifiedVision,
    UnifiedAudio,
    MiniCPM46,
    MuseGlimmer,
    Phi4,
    Gemma4Audio,
    Ocr,
    Ocr2
};

// ggml rope mode bits, as passed in the op_case.
constexpr int ROPE_NEOX = 1 << 16;
constexpr int ROPE_VISION = 3 << 16;

struct ProjectorDefinition {
    const char* modality;
    const char* name;
    EncoderTopology topology;
};

// Entries describe implemented graph topologies, independently of language DecoderConfig.
constexpr ProjectorDefinition projector_catalog[] = {
    {"vision", "deepseekocr", EncoderTopology::Ocr},
    {"vision", "deepseekocr2", EncoderTopology::Ocr2},
    {"vision", "pixtral", EncoderTopology::Pixtral},
    {"vision", "phi4", EncoderTopology::Phi4},
    {"vision", "muse-glimmer", EncoderTopology::MuseGlimmer},
    {"vision", "gemma4v", EncoderTopology::Gemma4},
    {"vision", "gemma4uv", EncoderTopology::UnifiedVision},
    {"audio", "gemma4a", EncoderTopology::Gemma4Audio},
    {"audio", "gemma4ua", EncoderTopology::UnifiedAudio},
    {"vision", "minicpmv4_6", EncoderTopology::MiniCPM46},
    {"vision", "gemma3", EncoderTopology::Siglip},
    {"vision", "idefics3", EncoderTopology::Siglip},
    {"vision", "janus_pro", EncoderTopology::Siglip},
    {"vision", "mlp", EncoderTopology::Clip},
    {"vision", "internvl", EncoderTopology::Internvl},
    {"vision", "resampler", EncoderTopology::Resampler},
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
    int64_t version = 0, queries = 0, kv_heads = 0;
    std::vector<int64_t> feature_layers;
    float eps;
    bool clip = false;  // clamp linear inputs/outputs to the recorded per-tensor bounds
    bool rms = false;   // encoder norms are RMS rather than layer norms
};

bool rms_encoder(const EncoderConfig& c) {
    return c.projector == "qwen2.5vl_merger" || c.topology == EncoderTopology::Pixtral ||
           c.topology == EncoderTopology::Gemma4 || c.topology == EncoderTopology::Ocr2 ||
           (c.topology == EncoderTopology::Internvl && c.width == 3200 && c.layers == 45);
}

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
    c.clip = c.topology == EncoderTopology::Gemma4 || c.topology == EncoderTopology::Gemma4Audio;
    if (c.topology == EncoderTopology::UnifiedAudio) {
        c.width = 640;
        c.heads = 1;
        c.layers = 0;
        c.eps = 1e-6f;
        return c;
    }
    c.width = positive(meta, key + "embedding_length");
    c.heads = c.topology == EncoderTopology::UnifiedVision ? 1 : positive(meta, key + "attention.head_count");
    c.kv_heads = c.topology == EncoderTopology::Ocr2 ? positive(meta, key + "attention.head_count_kv") : c.heads;
    c.layers = c.topology == EncoderTopology::UnifiedVision ? 0 : positive(meta, key + "block_count");
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
    c.eps = c.topology == EncoderTopology::Gemma4Audio ? 1e-6f : static_cast<float>(*eps);
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
        if (c.topology == EncoderTopology::Ocr || c.topology == EncoderTopology::Ocr2) {
            c.patch = 16;
            c.merge = 4;
            if (c.topology == EncoderTopology::Ocr)
                c.eps = 1e-5f;
            else
                c.activation = "GGML_UNARY_OP_SILU";
            return c;
        }
        if (c.topology == EncoderTopology::Pixtral || c.topology == EncoderTopology::Gemma4 ||
            c.topology == EncoderTopology::UnifiedVision || c.topology == EncoderTopology::MiniCPM46 ||
            c.topology == EncoderTopology::Phi4) {
            if (c.topology == EncoderTopology::Pixtral)
                c.merge = meta.get_int(key + "spatial_merge_size").value_or(1);
            else if (c.topology != EncoderTopology::Phi4)
                c.merge = meta.get_int(key + "projector.scale_factor")
                              .value_or(c.topology == EncoderTopology::MiniCPM46 ? 4 : 3);
            OPENVINO_ASSERT(c.merge > 0, "[GGUF] invalid vision merge size");
            if (c.topology == EncoderTopology::UnifiedVision) {
                c.patch *= c.merge;
                c.merge = 1;
            }
            if (c.topology == EncoderTopology::MiniCPM46) {
                auto layers = meta.get_int_array(key + "wa_layer_indexes");
                c.window_pattern = layers.empty() ? 0 : layers.front();
                OPENVINO_ASSERT(c.merge == 4 && c.window_pattern >= 0 && c.window_pattern < c.layers,
                                "[GGUF] invalid minicpmv4_6 merger configuration");
            }
            return c;
        }
        if (c.topology == EncoderTopology::MuseGlimmer) {
            c.merge = meta.get_int(key + "spatial_merge_size").value_or(2);
            c.window_pattern = 4;
            c.activation = "GGML_UNARY_OP_GELU_ERF";
            OPENVINO_ASSERT(c.merge > 0 && c.width / c.heads % 4 == 0,
                            "[GGUF] invalid Muse Glimmer merge or rotary head dimensions");
            return c;
        }
        if (c.topology == EncoderTopology::Resampler) {
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
            if (!ctx.metadata.get_bool(std::string("clip.has_") + modality + "_encoder").value_or(false))
                continue;
            auto c = config(ctx.metadata, modality);
            c.rms = rms_encoder(c);
            encoders.push_back(std::move(c));
        }
        OPENVINO_ASSERT(!encoders.empty(), "[GGUF] mmproj has no encoder");
        for (const auto& c : encoders) {
            clippable = c.clip;
            auto output = c.modality == "vision" ? vision(c) : audio(c);
            g.set_output(output, c.modality + ".embeddings");
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
        graph->mmproj_config["version"] = std::string("1");
        for (const auto& c : encoders) {
            graph->mmproj_config[c.modality + ".projector"] = c.projector;
            graph->mmproj_config[c.modality + ".output"] = c.modality + ".embeddings";
            graph->mmproj_config[c.modality + ".output_layout"] = std::string("1,B,T,D");
            graph->mmproj_config[c.modality + ".merge"] = std::to_string(c.merge);
            if (c.topology == EncoderTopology::MuseGlimmer)
                graph->mmproj_config["vision.window_size"] = std::to_string(vision_window_size);
            if (c.topology == EncoderTopology::Resampler) {
                graph->mmproj_config["vision.minicpmv_version"] = std::to_string(c.version);
                graph->mmproj_config["vision.query_count"] = std::to_string(c.queries);
            }
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
    GgufValue default_clip_min, default_clip_max;
    // Set once per encoder in build(); the Gemma4 families clamp every linear's input and output.
    bool clippable = false;
    int64_t vision_window_size = 0;

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
    // clip_input=false when the caller already clamped x to this projection's input bounds.
    GgufValue linear(const GgufValue& x, const std::string& base, bool with_bias = true, bool clip_input = true) {
        auto y = g.node(
            "GGML_OP_MUL_MAT",
            {g.tensors().require(base + ".weight"), clippable && clip_input ? clip_linear(x, base, "input") : x});
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
        return g.node("GGML_OP_FLASH_ATTN_EXT", inputs, 0, {{"encoder_attention", true}, {"scale", factor}});
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
        // The pinned ggml CLAMP aliases its source; the following gate observes the clipped input.
        auto input = clippable ? clip_linear(x, up, "input") : x;
        auto y = linear(input, up, with_bias, false);
        if (!gate.empty() && g.tensors().has(gate + ".weight"))
            y = mul(y, g.node(activation, {linear(input, gate)}));
        else
            y = g.node(activation, {y});
        return linear(y, down, with_bias);
    }
    GgufValue pool(const GgufValue& x, int64_t kx, int64_t ky) {
        return g.node("GGML_OP_POOL_2D", {x}, 0, {{"pool_params", std::vector<int64_t>{1, kx, ky, kx, ky, 0, 0}}});
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
        // Families whose ffn_up/ffn_down tensor names are swapped relative to their roles; the
        // per-layer width check below confirms it.
        const bool swappable = c.projector == "gemma3" || c.projector == "idefics3" ||
                               c.topology == EncoderTopology::Clip || c.projector == "qwen2vl_merger" ||
                               c.projector == "qwen2.5vl_merger";
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
                // ggml CLAMP is in-place: Q input clipping carries into K, then V.
                if (clippable && !fused)
                    z = clip_linear(z, p + name, "input");
                auto value = fused ? slice(qkv, 3, offset * c.width, c.width) : linear(z, p + name, true, false);
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
            // GQA head expansion is done by the FLASH_ATTN_EXT translator.
            OPENVINO_ASSERT(c.heads % c.kv_heads == 0, "[GGUF] encoder GQA head count mismatch");
            if (rope_positions) {
                q = encoder_rope(q, c, rope_positions, rope_positions_b);
                k = encoder_rope(k, c, rope_positions, rope_positions_b);
            }
            if (c.topology == EncoderTopology::Gemma4)
                v = g.build_norm(v, {}, c.eps);
            const auto mask = window_mask && (c.window_pattern == 0 || (i + 1) % c.window_pattern != 0) &&
                                      !(c.topology == EncoderTopology::MuseGlimmer && i == c.layers - 1)
                                  ? window_mask
                                  : GgufValue{};
            z = attention(q,
                          k,
                          v,
                          c.topology == EncoderTopology::Gemma4 ? 1.f : 1.f / std::sqrt(float(c.width / c.heads)),
                          mask);
            z = linear(reshape(z, {0, 1, -1, c.width}, true), p + "attn_out");
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
        if (c.projector == "qwen2a" || c.projector == "voxtral" || c.projector == "musicflamingo")
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
        if (c.topology == EncoderTopology::Ocr2) {
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
        const bool gemma4 = c.topology == EncoderTopology::Gemma4;
        const int mode = gemma4 ? ROPE_NEOX : 0;
        r.freq_base = gemma4 ? 100.f : 10000.f;
        auto a = rope(slice(x, 3, 0, r.n_dims), positions, mode);
        if (c.topology == EncoderTopology::Pixtral)
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
        if (c.topology == EncoderTopology::Ocr || c.topology == EncoderTopology::Ocr2)
            return ocr_vision(c);
        if (c.topology == EncoderTopology::Pixtral || c.topology == EncoderTopology::Gemma4 ||
            c.topology == EncoderTopology::UnifiedVision || c.topology == EncoderTopology::Phi4)
            return dynamic_vision(c);
        if (c.topology == EncoderTopology::MiniCPM46)
            return minicpm46(c);
        if (c.topology == EncoderTopology::Qwen)
            return qwen_vision(c);
        if (c.topology == EncoderTopology::MuseGlimmer)
            return muse_glimmer_vision(c);
        if (c.topology == EncoderTopology::Resampler)
            return resampler_vision(c);
        auto x = g.add_input("vision.pixel_values", ov::element::f32, {1, 3, c.image_size, c.image_size});
        x = patch_embeddings(convolution(x, "v.patch_embd.weight", c.patch), c.width);
        if ((c.topology == EncoderTopology::Clip || c.topology == EncoderTopology::Internvl) &&
            g.tensors().has("v.class_embd"))
            x = concat(x, reshape(g.tensors().require("v.class_embd"), {1, 1, 1, c.width}), 1);
        x = vit(x, c, g.tensors().require("v.position_embd.weight"));
        const auto side = c.image_size / c.patch;
        if (c.topology == EncoderTopology::Internvl) {
            OPENVINO_ASSERT(g.tensors().has("v.class_embd"), "[GGUF] InternVL requires a class embedding");
            x = slice(x, 2, 0, side * side);
            x = reshape(x, {1, side, side / c.merge, c.width * c.merge});
            x = transpose(x, {0, 2, 1, 3});
            x = reshape(x, {1, side / c.merge, side / c.merge, c.width * c.merge * c.merge});
            x = transpose(x, {0, 2, 1, 3});
            x = reshape(x, {1, 1, -1, c.width * c.merge * c.merge});
            return ffn(norm(x, "mm.model.mlp.0", 1e-5f), "mm.model.mlp.1", "mm.model.mlp.3", "GGML_UNARY_OP_GELU");
        }
        if (c.topology == EncoderTopology::Clip) {
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
    GgufValue convolution(const GgufValue& pixels, const std::string& weight, int64_t stride, int64_t padding = 0) {
        return g.node("GGML_OP_CONV_2D",
                      {g.tensors().require(weight), pixels},
                      0,
                      {{"conv_params", std::vector<int64_t>{stride, stride, padding, padding, 1, 1}}});
    }
    GgufValue unfold(const GgufValue& x, int64_t channels, int64_t kernel) {
        ov::Tensor dummy(ov::element::f32, {1, size_t(channels), size_t(kernel), size_t(kernel)});
        std::fill_n(dummy.data<float>(), dummy.get_size(), 0.f);
        return g.node("GGML_OP_IM2COL",
                      {g.add_constant("vision.unfold_kernel", dummy), x},
                      0,
                      {{"im2col_params", std::vector<int32_t>{int32_t(kernel), int32_t(kernel), 0, 0, 1, 1, 1}},
                       {"dst_type", ov::element::f32}});
    }
    GgufValue dynamic_vision(const EncoderConfig& c) {
        auto pixels = g.add_input("vision.pixel_values", ov::element::f32, {1, 3, -1, -1});
        GgufValue spatial, x;
        if (c.topology == EncoderTopology::UnifiedVision) {
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
            x = norm(linear(x, "v.patch_embd"), "v.patch_norm.2", 1e-5f);
        } else {
            spatial = convolution(c.topology == EncoderTopology::Gemma4 ? scale(pixels, 2.f, -1.f) : pixels,
                                  "v.patch_embd.weight",
                                  c.patch);
            x = patch_embeddings(spatial, c.width, c.topology != EncoderTopology::Gemma4);
        }
        GgufValue pos_a, pos_b, learned;
        if (c.topology == EncoderTopology::Phi4) {
            auto table =
                resize_square_table(g.tensors().require("v.position_embd.weight"), spatial, c.width, 0x201, "phi4");
            learned = reshape(transpose(table, {0, 2, 3, 1}), {1, 1, -1, c.width});
        } else {
            pos_a = index_input("position_x");
            pos_b = index_input("position_y");
            if (c.topology != EncoderTopology::Pixtral) {
                auto table = g.tensors().require("v.position_embd.weight");
                OPENVINO_ASSERT(table.ne(2) == 2, "[GGUF] Gemma4 position table requires x/y axes");
                table = reshape(table, {1, 2, table.ne(1), c.width});
                learned = add(g.node("GGML_OP_GET_ROWS", {slice(table, 1, 0, 1), pos_a}),
                              g.node("GGML_OP_GET_ROWS", {slice(table, 1, 1, 1), pos_b}));
            }
        }
        if (c.topology == EncoderTopology::UnifiedVision)
            return linear(g.build_norm(norm(add(x, learned), "v.patch_norm.3", 1e-5f), {}, c.eps),
                          "mm.input_projection");
        // Pixtral uses height then width, with alternating frequencies; Gemma4 uses x then y, NEOX halves.
        x = vit(x,
                c,
                learned,
                c.topology == EncoderTopology::Pixtral ? pos_b : pos_a,
                {},
                c.topology == EncoderTopology::Pixtral ? pos_a : pos_b);
        if (c.topology == EncoderTopology::Phi4) {
            g.tensors().require("mm.0.bias");
            g.tensors().require("mm.2.bias");
            return ffn(x, "mm.0", "mm.2", "GGML_UNARY_OP_GELU");
        }
        if (c.topology == EncoderTopology::Gemma4) {
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
        auto parts = gather4(x, "vit_merger");
        auto cat = concat(concat(concat(parts[0], parts[1]), parts[2]), parts[3]);
        x = add(ffn(norm(cat, p + "ds_ln", c.eps), p + "ds_ffn_up", p + "ds_ffn_down", "GGML_UNARY_OP_GELU"),
                scale(add(add(add(parts[0], parts[1]), parts[2]), parts[3]), .25f));
        x = vit(x, c, {}, {}, {}, {}, c.window_pattern + 1);
        parts = gather4(x, "merger");
        cat = concat(concat(concat(parts[0], parts[1]), parts[2]), parts[3]);
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
        vision_window_size = int64_t(std::sqrt(double(table.ne(1))));
        table = resize_square_table(table, spatial, c.width, 1, "Muse Glimmer");
        x = add(x, reshape(transpose(table, {0, 2, 3, 1}), {1, 1, -1, c.width}));
        x = gather_rows(x, "patch_indices");
        auto pos_x = index_input("position_x"), pos_y = index_input("position_y");
        auto mask = g.add_input("vision.attention_mask", ov::element::f32, {1, 1, -1, -1});
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
        auto patches = add(convolution(slice(pixels, 0, 0, 1), "v.patch_embd.weight", c.patch),
                           convolution(slice(pixels, 0, 1, 1), "v.patch_embd.weight.1", c.patch));
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
        const auto width = positive(ctx.metadata, "clip.vision.sam.embedding_length");
        const auto heads = positive(ctx.metadata, "clip.vision.sam.head_count");
        const auto layers = positive(ctx.metadata, "clip.vision.sam.block_count");
        const auto window = positive(ctx.metadata, "clip.vision.window_size");
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
            auto rw = g.node("GGML_OP_GET_REL_POS",
                             {g.tensors().require(p + "attn.pos_w.weight"), global_layer ? global : local},
                             1);
            auto rh = g.node("GGML_OP_GET_REL_POS",
                             {g.tensors().require(p + "attn.pos_h.weight"), global_layer ? global : local},
                             1);
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
        auto x = reshape(transpose(spatial, {0, 2, 3, 1}), {0, 1, -1, c.width}, true);
        if (c.topology == EncoderTopology::Ocr2) {
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
            EncoderConfig clip = c;
            clip.activation = "GGML_UNARY_OP_GELU_QUICK";
            x = vit(x, clip, pos);
            x = slice(x, 2, 1, std::numeric_limits<int32_t>::max() - 1);
            auto sam_features = reshape(transpose(spatial, {0, 2, 3, 1}), {0, 1, -1, c.width}, true);
            x = concat(x, sam_features);
        }
        x = linear(x, "mm.model.fc");
        const auto width = g.tensors().require("mm.model.fc.weight").ne(1);
        x = reshape(x, {1, 1, -1, width});
        // Indexing describes overview/tile row assembly, including learned separators.
        if (c.topology == EncoderTopology::Ocr)
            x = concat(x, reshape(g.tensors().require("v.image_newline"), {1, 1, 1, width}), 1);
        x = concat(x, reshape(g.tensors().require("v.view_seperator"), {1, 1, 1, width}), 1);
        return gather_rows(x, "output_indices");
    }
    GgufValue gemma4_audio(const EncoderConfig& c) {
        const auto mel = positive(ctx.metadata, "clip.audio.num_mel_bins");
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
        auto positions = g.add_input("audio.position_embeddings", ov::element::f32, {1, 1, 13, c.width});
        auto mask = g.add_input("audio.attention_mask", ov::element::f32, {1, 1, -1, -1});
        auto relative = g.add_input("audio.relative_indices", ov::element::i32, {1, 1, -1, -1});
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
            // ggml CLAMP is in-place: Q input clipping carries into K, then V.
            z = clip_linear(z, p + "attn_q", "input");
            auto q = linear(z, p + "attn_q", false, false);
            z = clip_linear(z, p + "attn_k", "input");
            auto k = linear(z, p + "attn_k", false, false);
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
        if (c.topology == EncoderTopology::Gemma4Audio)
            return gemma4_audio(c);
        if (c.topology == EncoderTopology::UnifiedAudio) {
            auto x = g.add_input("audio.waveform_frames", ov::element::f32, {1, 1, -1, 640});
            return linear(g.build_norm(x, {}, c.eps), "mm.a.input_projection");
        }
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
            x = concat(reshape(g.tensors().require("v.boi"), {1, 1, 1, width}), x, 1);
            return concat(x, reshape(g.tensors().require("v.eoi"), {1, 1, 1, width}), 1);
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
