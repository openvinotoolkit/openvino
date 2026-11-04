// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "projector_registry.hpp"

#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {

std::string resolve_projector_type(const GgufMetadata& metadata, const std::string& modality) {
    auto type = metadata.get_str("clip.projector_type").value_or("");
    if (type.empty())
        type = metadata.get_str("clip." + modality + ".projector_type").value_or("");
    if (type == "qwen2.5o")
        type = modality == "vision" ? "qwen2.5vl_merger" : "qwen2a";
    return type;
}

ProjectorRegistry::ProjectorRegistry(std::vector<ProjectorDefinition> definitions) {
    for (auto& definition : definitions)
        add(std::move(definition));
}

void ProjectorRegistry::add(ProjectorDefinition definition, RegistrationMode mode) {
    OPENVINO_ASSERT(!definition.id.empty() && !definition.architecture.empty() &&
                        (definition.modality == "vision" || definition.modality == "audio") &&
                        !definition.projector_type.empty() && definition.build,
                    "[GGUF] invalid projector definition");
    const auto existing = m_definitions.find(definition.id);
    OPENVINO_ASSERT(mode == RegistrationMode::Add ? existing == m_definitions.end() : existing != m_definitions.end(),
                    "[GGUF] cannot ",
                    mode == RegistrationMode::Add ? "add duplicate" : "replace unknown",
                    " projector handler '",
                    definition.id,
                    "'");
    const auto id = definition.id;
    m_definitions[id] = std::make_shared<const ProjectorDefinition>(std::move(definition));
}

std::shared_ptr<const ProjectorDefinition> ProjectorRegistry::find(const GgufMetadata& metadata,
                                                                   const std::string& modality,
                                                                   const std::string& projector_type) const {
    std::shared_ptr<const ProjectorDefinition> found;
    const auto architecture = metadata.architecture();
    for (const auto& entry : m_definitions) {
        const auto& definition = entry.second;
        if (definition->architecture != architecture || definition->modality != modality ||
            definition->projector_type != projector_type || (definition->match && !definition->match(metadata)))
            continue;
        OPENVINO_ASSERT(!found,
                        "[GGUF] multiple projector handlers claim ",
                        modality,
                        " projector '",
                        projector_type,
                        "'");
        found = definition;
    }
    return found;
}

std::vector<ProjectorDefinition> ProjectorRegistry::supported_projectors() const {
    std::vector<ProjectorDefinition> definitions;
    for (const auto& entry : m_definitions)
        definitions.push_back(*entry.second);
    return definitions;
}

}  // namespace ov::frontend::gguf
