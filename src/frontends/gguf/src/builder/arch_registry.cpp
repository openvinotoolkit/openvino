// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The native builder's architecture accept list, and the one per-architecture property that
// cannot be derived from the GGUF tensor table (the RoPE mode).
//
// Adding a same-family architecture is a one-line change here and nothing else: the decoder
// builder derives everything else -- QK-norm, biases, fused QKV, MoE routing, SWA, soft-caps --
// from the tensor table and metadata. See docs/adding_an_architecture.md.

#include "arch_registry.hpp"

namespace ov {
namespace frontend {
namespace gguf {

bool arch_uses_neox_rope(const std::string& arch) {
    return arch == "qwen2" || arch == "qwen3" || arch == "phi3" || arch == "hunyuan-dense" || arch == "gpt-oss" ||
           arch == "gemma" || arch == "gemma2" || arch == "gemma3" || arch == "gemma4" || arch == "olmoe" ||
           // H2 2025 dense additions
           arch == "exaone4" || arch == "plamo3" || arch == "mellum" ||
           // H2 2025 MoE additions
           arch == "hunyuan-moe" || arch == "glm4moe" || arch == "bailingmoe2" || arch == "exaone-moe" ||
           arch == "minimax-m2" ||
           // H2 2025 VL backbone / other
           arch == "jais2" || arch == "deepseek2-ocr";
}

const std::set<std::string>& verified_archs() {
    static const std::set<std::string> archs = {
        "llama",  // llama-2 / llama-3
        "qwen2",  // qwen2 / qwen2.5
        "qwen3",
        "phi3",     // phi-3 (fused QKV)
        "minicpm",  // NORMAL rope + scalar scales
        "hunyuan-dense",
        "olmoe",     // OLMoE 1B-7B (MoE)
        "qwen3moe",  // Qwen3 MoE: same topology as olmoe
        // Qwen3.5/3.6 hybrid: Gated-DeltaNet linear attention on 3 of every 4 layers, full
        // attention with M-RoPE and an interleaved query+gate projection on the rest. Verified
        // token-exact against llama.cpp on Qwen3.5-0.8B-Q8_0 and Ternary-Bonsai-27B-Q2_g64.
        // GREEDY / BATCH 1 ONLY: the recurrent conv and delta states are not reordered by
        // beam_idx and have a static batch of 1, so beam search or batch > 1 fails at inference
        // (a Concat shape mismatch on the conv window) rather than producing wrong output.
        "qwen35",
        "gpt-oss",  // MoE + sinks + SWA
        "gemma",    // Gemma 2B / 7B
        "gemma2",   // Gemma 2: post-norms + attention soft-cap
        "gemma3",   // Gemma 3: post-norms + final logit soft-cap
        "gemma4",   // Gemma 4: SWA, per-layer embeddings, shared KV
    };
    return archs;
}

const std::set<std::string>& experimental_archs() {
    static const std::set<std::string> archs = {
        // H2 2025: dense
        "llama-embed",  // Bidirectional LLaMA (embedding, no causal mask)
        "exaone4",      // EXAONE 4.0: NEOX rope, post-norms (attn+ffn)
        "plamo3",       // PLaMo-3: NEOX rope, post-norms (attn+ffn)
        "smollm3",      // SmolLM3: NORMAL rope + SWA
        // H2 2025: MoE
        "hunyuan-moe",   // Hunyuan MoE: NEOX rope, MoE routing, QK-norm
        "glm4moe",       // GLM 4.5 MoE: NEOX rope, 1 dense lead layer, MoE + attn post-norm
        "exaone-moe",    // EXAONE MoE: NEOX rope, SWA + MoE, shared expert
        "minimax-m2",    // Minimax M2: NEOX rope, pure MoE
        "ernie4_5-moe",  // Ernie 4.5 MoE: NORMAL rope, dense lead layers + MoE stride
        "bailingmoe2",   // BailingMoe V2: NEOX rope, MoE + shared expert + QK-norm
        // 2026: dense
        "maincoder",     // Maincoder-1B: NORMAL rope, QK-norm (auto-detected)
        "mistral3",      // Ministral-3B: NORMAL rope, dense
        "muse-glimmer",  // Muse Glimmer (Meta Onyx): NORMAL rope on SWA layers only (global
                         // layers are NoPE), sigmoid attention output gate, QK-norm,
                         // pre+post norms, final logit soft-cap
        // 2026: MoE
        "mellum",         // JetBrains Mellum: NEOX rope, pure MoE
        "deepseek2-ocr",  // DeepSeekOCR: NEOX rope, dense lead layers + MoE
        "jais2",          // JAIS-2: NEOX rope, dense (biases auto-detected)
    };
    return archs;
}

const std::set<std::string>& supported_archs() {
    static const std::set<std::string> archs = [] {
        std::set<std::string> a = verified_archs();
        a.insert(experimental_archs().begin(), experimental_archs().end());
        return a;
    }();
    return archs;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
