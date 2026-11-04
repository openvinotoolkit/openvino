// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/extension/projector.hpp"

namespace ov::frontend::gguf {

std::vector<ProjectorDefinition> builtin_projectors();
std::string resolve_projector_type(const GgufMetadata& metadata, const std::string& modality);

class ProjectorRegistry {
public:
    explicit ProjectorRegistry(std::vector<ProjectorDefinition> definitions = builtin_projectors());
    void add(ProjectorDefinition definition, RegistrationMode mode = RegistrationMode::Add);
    std::shared_ptr<const ProjectorDefinition> find(const GgufMetadata& metadata,
                                                    const std::string& modality,
                                                    const std::string& projector_type) const;
    std::vector<ProjectorDefinition> supported_projectors() const;

private:
    std::map<std::string, std::shared_ptr<const ProjectorDefinition>> m_definitions;
};

}  // namespace ov::frontend::gguf
