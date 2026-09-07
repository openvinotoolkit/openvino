// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arch_registry.hpp"

#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {

namespace {
struct DecoderEntry {
    const char* name;
    RopeMode rope;
    Maturity maturity;
};
const DecoderEntry decoders[] = {
    {"bailingmoe2", RopeMode::Neox, Maturity::Experimental},
    {"deepseek2-ocr", RopeMode::Neox, Maturity::Experimental},
    {"ernie4_5-moe", RopeMode::Normal, Maturity::Experimental},
    {"exaone-moe", RopeMode::Neox, Maturity::Experimental},
    {"exaone4", RopeMode::Neox, Maturity::Experimental},
    {"gemma", RopeMode::Neox, Maturity::Experimental},
    {"gemma2", RopeMode::Neox, Maturity::Experimental},
    {"gemma3", RopeMode::Neox, Maturity::Verified},
    {"gemma4", RopeMode::Neox, Maturity::Verified},
    {"glm4moe", RopeMode::Neox, Maturity::Experimental},
    {"gpt-oss", RopeMode::Neox, Maturity::Verified},
    {"hunyuan-dense", RopeMode::Neox, Maturity::Verified},
    {"hunyuan-moe", RopeMode::Neox, Maturity::Experimental},
    {"jais2", RopeMode::Neox, Maturity::Experimental},
    {"llama", RopeMode::Normal, Maturity::Verified},
    {"llama-embed", RopeMode::Normal, Maturity::Experimental},
    {"maincoder", RopeMode::Normal, Maturity::Experimental},
    {"mellum", RopeMode::Neox, Maturity::Experimental},
    {"minicpm", RopeMode::Normal, Maturity::Verified},
    {"minimax-m2", RopeMode::Neox, Maturity::Experimental},
    {"mistral3", RopeMode::Normal, Maturity::Experimental},
    {"muse-glimmer", RopeMode::Normal, Maturity::Experimental},
    {"olmoe", RopeMode::Neox, Maturity::Verified},
    {"phi3", RopeMode::Neox, Maturity::Verified},
    {"plamo3", RopeMode::Neox, Maturity::Experimental},
    {"qwen2", RopeMode::Neox, Maturity::Verified},
    {"qwen3", RopeMode::Neox, Maturity::Verified},
    {"qwen35", RopeMode::Interleaved, Maturity::Verified},
    {"qwen3moe", RopeMode::Normal, Maturity::Experimental},
    {"smollm3", RopeMode::Normal, Maturity::Experimental},
};
}  // namespace

std::vector<ArchitectureDefinition> builtin_architectures() {
    std::vector<ArchitectureDefinition> definitions;
    for (const auto& entry : decoders) {
        definitions.push_back(make_decoder_architecture(entry.name, entry.rope, {}, entry.maturity));
    }
    // Add custom definitions here; their factories and builders are shared with external plugins.
    return definitions;
}

bool arch_uses_neox_rope(const std::string& arch) {
    for (const auto& entry : decoders) {
        if (arch == entry.name)
            return entry.rope == RopeMode::Neox;
    }
    return false;
}

const std::set<std::string>& verified_archs() {
    static const auto names = [] {
        std::set<std::string> result;
        for (const auto& entry : decoders)
            if (entry.maturity == Maturity::Verified)
                result.insert(entry.name);
        return result;
    }();
    return names;
}

const std::set<std::string>& experimental_archs() {
    static const auto names = [] {
        std::set<std::string> result;
        for (const auto& entry : decoders)
            if (entry.maturity == Maturity::Experimental)
                result.insert(entry.name);
        return result;
    }();
    return names;
}

const std::set<std::string>& supported_archs() {
    static const auto names = [] {
        auto result = verified_archs();
        result.insert(experimental_archs().begin(), experimental_archs().end());
        return result;
    }();
    return names;
}

ArchRegistry::ArchRegistry(std::vector<ArchitectureDefinition> definitions) {
    for (auto& definition : definitions)
        add(std::move(definition), RegistrationMode::Add);
}

void ArchRegistry::add(ArchitectureDefinition definition, RegistrationMode mode) {
    OPENVINO_ASSERT(!definition.id.empty() && !definition.architecture.empty(),
                    "[GGUF] architecture definition requires a handler id and architecture name");
    OPENVINO_ASSERT(definition.factory, "[GGUF] architecture '", definition.id, "' has no builder factory");
    const auto existing = m_definitions.find(definition.id);
    if (mode == RegistrationMode::Add) {
        OPENVINO_ASSERT(existing == m_definitions.end(),
                        "[GGUF] duplicate architecture handler '",
                        definition.id,
                        "'; use RegistrationMode::Replace to replace it explicitly");
    } else {
        OPENVINO_ASSERT(existing != m_definitions.end(),
                        "[GGUF] cannot replace unknown architecture handler '",
                        definition.id,
                        "'");
    }
    const auto id = definition.id;
    m_definitions[id] = std::make_shared<const ArchitectureDefinition>(std::move(definition));
}

void ArchRegistry::add_extension(const ArchitectureExtension::Ptr& ext) {
    OPENVINO_ASSERT(ext, "[GGUF] null ArchitectureExtension");
    add(ext->definition(), ext->registration_mode());
}

std::shared_ptr<const ArchitectureDefinition> ArchRegistry::find(const GgufMetadata& meta) const {
    std::shared_ptr<const ArchitectureDefinition> found;
    for (const auto& [id, definition] : m_definitions) {
        if (!definition->matches(meta))
            continue;
        OPENVINO_ASSERT(!found, "[GGUF] architecture handlers '", found->id, "' and '", id, "' both claim this file");
        found = definition;
    }
    return found;
}

std::string ArchRegistry::describe_supported() const {
    std::string out;
    for (const auto& [id, definition] : m_definitions) {
        out += (out.empty() ? "" : ", ") + id;
    }
    return out;
}

const ArchRegistry& default_arch_registry() {
    static const ArchRegistry registry;
    return registry;
}

}  // namespace ov::frontend::gguf
