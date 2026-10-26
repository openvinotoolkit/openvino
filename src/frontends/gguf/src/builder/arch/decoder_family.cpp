// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <limits>

#include "builder/arch_registry.hpp"
#include "builder/decoder_metadata.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {

DecoderParameters decoder_family_defaults(const std::string& arch, int layers, int embedding, int heads) {
    DecoderParameters p;
    p.architecture = arch;
    p.attention.rope_mode = arch_uses_neox_rope(arch) ? RopeMode::Neox : RopeMode::Normal;
    p.embedding_model = arch == "llama-embed";
    if (arch == "jais2") {
        p.normalization.kind = NormalizationKind::Layer;
        p.activation = FfnActivation::ReluSquared;
    }
    const bool gemma = arch == "gemma" || arch == "gemma2" || arch == "gemma3" || arch == "gemma4";
    if (gemma) {
        p.activation = FfnActivation::GeGlu;
        p.scales.embedding = std::sqrt(static_cast<float>(embedding));
    }
    if (arch.rfind("minicpm", 0) == 0) {
        p.scales.embedding = 12.0f;
        p.scales.residual = 1.4f / std::sqrt(static_cast<float>(layers));
        p.scales.logits = 256.0f / static_cast<float>(embedding);
    }
    p.normalization.qk_after_rope = arch == "hunyuan-dense" || arch == "hunyuan-moe" || arch == "maincoder";
    p.normalization.post_only = arch == "exaone4";
    p.normalization.value_norm = arch == "gemma4";
    if (arch == "gemma4") {
        p.normalization.qk_full_width = false;
        p.scales.attention = 1.0f;
    } else if (arch == "gemma2" && layers == 46 && heads > 0) {
        p.scales.attention = 1.0f / std::sqrt(static_cast<float>(embedding / heads));
    }
    p.scales.token_embeddings_only = arch == "gemma3" || arch == "gemma4";
    p.attention.sliding_window = arch == "gemma3" || arch == "gemma4";
    p.attention.window_size = (arch == "gemma2" || (arch == "exaone4" && layers == 64)) ? 4096 : 0;
    p.attention.sliding_period = arch == "gemma3"                              ? 6
                                 : arch == "plamo3"                            ? 8
                                 : (arch == "exaone4" || arch == "exaone-moe") ? 4
                                                                               : 2;
    if (arch == "gemma3" || arch == "plamo3")
        p.attention.sliding_rope.freq_base = 10000.0f;
    else
        p.attention.sliding_rope.freq_base = 0.0f;
    // Zero inherits the global scale; local Gemma RoPE retains its training scale.
    p.attention.sliding_rope.freq_scale = arch == "gemma3" || arch == "gemma4" ? 1.0f : 0.0f;
    p.normalization.embedding_norm = arch == "muse-glimmer";
    p.normalization.post_epsilon = arch == "muse-glimmer" ? 1e-8f : 0.0f;
    p.attention.rope_on_sliding_only =
        arch == "muse-glimmer" || arch == "exaone-moe" || (arch == "exaone4" && layers == 64);
    p.attention.rope_skip_period = arch == "smollm3" ? 4 : 0;
    p.experts.normalize_weights = arch == "hunyuan-moe" || arch == "minimax-m2" || arch == "qwen3moe" ||
                                  arch == "qwen35moe" || arch == "ernie4_5-moe" || arch == "mellum" || arch == "gemma4";
    if (arch == "glm4moe")
        p.experts.routing = ExpertRouting::Sigmoid;
    if (arch == "qwen35" || arch == "qwen35moe") {
        p.hybrid.kind = RecurrentKind::GatedDeltaNet;
        p.attention.rope_mode = RopeMode::Interleaved;
        p.attention.interleaved_q_gate = true;
        p.attention.fused_qkv = false;
        p.attention.output_gate = true;
        p.normalization.qk_norm = true;
        p.normalization.qk_full_width = false;
        p.model_contract.batchable_recurrent = true;
    }
    if (arch == "lfm2" || arch == "lfm2moe") {
        p.attention.sliding_on_attention_layers = true;
        p.model_contract.batchable_recurrent = true;
    }
    if (arch == "gemma3")
        p.model_contract.image_attention = ModelContract::ImageAttentionMode::GlobalAndSliding;
    if (arch == "gemma4" && embedding != 1536 && embedding != 2560)
        p.model_contract.image_attention = ModelContract::ImageAttentionMode::SlidingOnly;
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
    if (p.architecture.rfind("minicpm", 0) == 0 && p.scales.logits != 0.0f)
        p.scales.logits = 1.0f / p.scales.logits;
    if (p.architecture == "glm4moe" && static_cast<int>(p.experts.routing) == 0)
        p.experts.routing = ExpertRouting::Sigmoid;
    if (p.architecture == "lfm2moe")
        p.experts.weights_scale = 0.0f;
    apply_decoder_options(p, options);
    return p;
}

}  // namespace ov::frontend::gguf
