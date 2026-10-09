// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arch_registry.hpp"

#include <optional>
#include <utility>

#include "builder/arch/mamba_builder.hpp"
#include "builder/arch/mmproj_builder.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {

namespace {
struct ArchitectureEntry {
    ArchitectureEntry(const char* name, RopeMode rope) : name(name), rope(rope) {}

    ArchitectureEntry(const char* name,
                      ArchitectureDefinition::BuilderFactory factory,
                      ArchitectureDefinition::MatchFn match = {},
                      const char* id = nullptr)
        : name(name),
          factory(std::move(factory)),
          match(std::move(match)),
          id(id) {}

    const char* name;
    std::optional<RopeMode> rope;
    ArchitectureDefinition::BuilderFactory factory = {};
    ArchitectureDefinition::MatchFn match = {};
    const char* id = nullptr;
};
const ArchitectureEntry architectures[] = {
    {"bailingmoe2", RopeMode::Neox},
    {"clip",
     make_mmproj_builder,
     [](const GgufMetadata& meta) {
         return meta.has("clip.projector_type") || meta.has("clip.vision.projector_type") ||
                meta.has("clip.audio.projector_type");
     },
     "clip.mmproj"},
    {"deepseek2-ocr", RopeMode::Neox},
    {"ernie4_5-moe", RopeMode::Normal},
    {"exaone-moe", RopeMode::Neox},
    {"exaone4", RopeMode::Neox},
    {"gemma", RopeMode::Neox},
    {"gemma2", RopeMode::Neox},
    {"gemma3", RopeMode::Neox},
    {"gemma4", RopeMode::Neox},
    {"glm4moe", RopeMode::Neox},
    {"gpt-oss", RopeMode::Neox},
    {"hunyuan-dense", RopeMode::Neox},
    {"hunyuan-moe", RopeMode::Neox},
    {"jais2", RopeMode::Neox},
    {"llama", RopeMode::Normal},
    {"llama-embed", RopeMode::Normal},
    {"maincoder", RopeMode::Normal},
    {"mamba2", make_mamba2_builder},
    {"mellum", RopeMode::Neox},
    {"minicpm", RopeMode::Normal},
    {"minimax-m2", RopeMode::Neox},
    {"mistral3", RopeMode::Normal},
    {"muse-glimmer", RopeMode::Normal},
    {"nemotron_h", make_mamba2_builder},
    {"olmoe", RopeMode::Neox},
    {"phi3", RopeMode::Neox},
    {"plamo3", RopeMode::Neox},
    {"qwen2", RopeMode::Neox},
    {"qwen3", RopeMode::Neox},
    {"qwen35", RopeMode::Interleaved},
    {"qwen35moe", RopeMode::Interleaved},
    {"qwen3moe", RopeMode::Neox},
    {"smollm3", RopeMode::Normal},
};
}  // namespace

std::vector<ArchitectureDefinition> builtin_architectures() {
    std::vector<ArchitectureDefinition> definitions;
    for (const auto& entry : architectures) {
        if (entry.factory) {
            definitions.push_back({entry.id ? entry.id : entry.name, entry.name, entry.factory, entry.match});
        } else {
            definitions.push_back(make_decoder_architecture(entry.name, entry.rope.value()));
        }
    }
    return definitions;
}

bool arch_uses_neox_rope(const std::string& arch) {
    for (const auto& entry : architectures) {
        if (arch == entry.name)
            return entry.rope == RopeMode::Neox;
    }
    return false;
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
    if (const auto projector = std::dynamic_pointer_cast<ProjectorExtension>(ext))
        m_projectors.add(projector->projector_definition(), projector->registration_mode());
    else
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

std::set<std::string> ArchRegistry::supported_archs() const {
    std::set<std::string> names;
    for (const auto& entry : m_definitions)
        names.insert(entry.second->architecture);
    return names;
}

std::string ArchRegistry::describe_supported() const {
    std::string out;
    for (const auto& architecture : supported_archs())
        out += (out.empty() ? "" : ", ") + architecture;
    return out;
}

const ArchRegistry& default_arch_registry() {
    static const ArchRegistry registry;
    return registry;
}

}  // namespace ov::frontend::gguf
