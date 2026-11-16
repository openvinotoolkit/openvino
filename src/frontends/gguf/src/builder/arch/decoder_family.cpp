// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>

#include "builder/arch_registry.hpp"
#include "builder/decoder_metadata.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {
namespace {
bool is_one_of(const std::string& arch, std::initializer_list<const char*> names) {
    return std::any_of(names.begin(), names.end(), [&arch](const char* name) {
        return arch == name;
    });
}
}  // namespace

DecoderParameters decoder_family_defaults(const std::string& arch, int layers, int embedding, int heads) {
    DecoderParameters p;
    p.architecture = arch;
    p.attention.rope_mode = arch_uses_neox_rope(arch) ? RopeMode::Neox : RopeMode::Normal;
    p.embedding_model = arch == "llama-embed";
    if (arch == "jais2") {
        p.normalization.kind = NormalizationKind::Layer;
        p.activation = FfnActivation::ReluSquared;
    }
    if (is_one_of(arch, {"gemma", "gemma2", "gemma3", "gemma4"})) {
        p.activation = FfnActivation::GeGlu;
        p.scales.embedding = std::sqrt(static_cast<float>(embedding));
    }
    if (arch == "afmoe")
        p.scales.embedding = std::sqrt(static_cast<float>(embedding));
    if (arch.rfind("minicpm", 0) == 0) {
        p.scales.embedding = 12.0f;
        p.scales.residual = 1.4f / std::sqrt(static_cast<float>(layers));
        // logit_scale is a divisor; correct upstream's inverted missing-key default.
        p.scales.logits = static_cast<float>(embedding) / 256.0f;
    }
    p.normalization.qk_after_rope = is_one_of(arch, {"hunyuan-dense", "hunyuan-moe", "maincoder"});
    p.normalization.post_only = arch == "exaone4";
    p.normalization.value_norm = arch == "gemma4";
    if (arch == "gemma4") {
        p.normalization.qk_full_width = false;
        p.scales.attention = 1.0f;
    } else if (arch == "gemma2" && layers == 46 && heads > 0) {
        p.scales.attention = 1.0f / std::sqrt(static_cast<float>(embedding / heads));
    }
    p.scales.token_embeddings_only = is_one_of(arch, {"gemma3", "gemma4"});
    p.attention.sliding_window = p.scales.token_embeddings_only;
    p.attention.window_size = (arch == "gemma2" || (arch == "exaone4" && layers == 64)) ? 4096 : 0;
    p.attention.sliding_period = arch == "gemma3"                                                 ? 6
                                 : arch == "plamo3"                                               ? 8
                                 : (arch == "afmoe" || arch == "exaone4" || arch == "exaone-moe") ? 4
                                                                                                  : 2;
    if (is_one_of(arch, {"gemma3", "plamo3"}))
        p.attention.sliding_rope.freq_base = 10000.0f;
    // Zero inherits the global scale; local Gemma RoPE retains its training scale.
    if (is_one_of(arch, {"gemma3", "gemma4"}))
        p.attention.sliding_rope.freq_scale = 1.0f;
    if (arch == "muse-glimmer") {
        p.normalization.embedding_norm = true;
        p.normalization.post_epsilon = 1e-8f;
    }
    p.attention.rope_on_sliding_only =
        arch == "muse-glimmer" || arch == "exaone-moe" || (arch == "exaone4" && layers == 64);
    if (is_one_of(arch, {"afmoe", "smollm3"}))
        p.attention.rope_skip_period = 4;
    p.experts.normalize_weights =
        is_one_of(arch, {"hunyuan-moe", "minimax-m2", "qwen3moe", "qwen35moe", "ernie4_5-moe", "mellum", "gemma4"});
    if (is_one_of(arch, {"afmoe", "glm4moe"}))
        p.experts.routing = ExpertRouting::Sigmoid;
    if (is_one_of(arch, {"qwen35", "qwen35moe"})) {
        p.hybrid.kind = RecurrentKind::GatedDeltaNet;
        p.attention.rope_mode = RopeMode::Interleaved;
        p.attention.interleaved_q_gate = true;
        p.attention.fused_qkv = false;
        p.attention.output_gate = true;
        p.normalization.qk_norm = true;
        p.normalization.qk_full_width = false;
        p.model_contract.batchable_recurrent = true;
    }
    if (is_one_of(arch, {"lfm2", "lfm2moe"})) {
        p.attention.sliding_on_attention_layers = true;
        p.model_contract.batchable_recurrent = true;
    }
    if (arch == "gemma3")
        p.model_contract.image_attention = ModelContract::ImageAttentionMode::GLOBAL_AND_SLIDING;
    if (arch == "gemma4" && embedding != 1536 && embedding != 2560)
        p.model_contract.image_attention = ModelContract::ImageAttentionMode::SLIDING_ONLY;
    return p;
}

DecoderParameters resolve_decoder_parameters(const GgufMetadata& metadata,
                                             RopeMode rope,
                                             const DecoderOptions& options) {
    const auto architecture = metadata.get_str("general.architecture");
    OPENVINO_ASSERT(architecture, "[GGUF] missing or non-string general.architecture");
    const auto dimension = [&](const std::string& suffix) {
        auto value = metadata.get_int(*architecture + "." + suffix);
        if (!value && suffix == "attention.head_count") {
            const auto heads = metadata.get_int_array(*architecture + "." + suffix);
            if (!heads.empty())
                value = *std::max_element(heads.begin(), heads.end());
        }
        OPENVINO_ASSERT(value && *value > 0 && *value <= std::numeric_limits<int>::max(),
                        "[GGUF] invalid required decoder dimension '",
                        *architecture,
                        ".",
                        suffix,
                        "'");
        return static_cast<int>(*value);
    };
    auto defaults = decoder_family_defaults(*architecture,
                                            dimension("block_count"),
                                            dimension("embedding_length"),
                                            dimension("attention.head_count"));
    defaults.attention.rope_mode = rope;
    if (options.normalization)
        defaults.normalization.kind = *options.normalization;
    if (options.norm_epsilon)
        defaults.normalization.epsilon = *options.norm_epsilon;
    auto p = read_decoder_parameters(metadata, defaults);
    // llama.cpp always renormalizes top-K weights for these built-in families.
    p.experts.normalize_weights = defaults.experts.normalize_weights || p.experts.normalize_weights;
    if (p.architecture.rfind("minicpm", 0) == 0 && p.scales.logits != 0.0f)
        p.scales.logits = 1.0f / p.scales.logits;
    if (p.architecture == "lfm2moe")
        p.experts.weights_scale = 0.0f;
    apply_decoder_options(p, options);
    return p;
}

}  // namespace ov::frontend::gguf
