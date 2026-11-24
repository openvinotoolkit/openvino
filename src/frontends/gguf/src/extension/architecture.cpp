// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/extension/architecture.hpp"

#include "builder/decoder_metadata.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {

bool ArchitectureDefinition::matches(const GgufMetadata& metadata) const {
    return metadata.architecture() == architecture && (!match || match(metadata));
}

ArchitectureDefinition make_decoder_architecture(std::string architecture, RopeMode rope, DecoderOptionsFn options) {
    ArchitectureDefinition definition;
    definition.id = definition.architecture = std::move(architecture);
    definition.factory = decoder_builder_factory(rope, std::move(options));
    return definition;
}

ArchitectureDefinition make_decoder_architecture(std::string architecture, DecoderParametersFn parameters) {
    ArchitectureDefinition definition;
    definition.id = definition.architecture = std::move(architecture);
    definition.factory = decoder_builder_factory(std::move(parameters));
    return definition;
}

ArchitectureExtension::ArchitectureExtension(ArchitectureDefinition definition, RegistrationMode mode)
    : m_definition(std::move(definition)),
      m_mode(mode) {}

ArchitectureExtension::ArchitectureExtension(std::string architecture, RopeMode rope)
    : ArchitectureExtension(make_decoder_architecture(std::move(architecture), rope)) {}

ArchitectureExtension::ArchitectureExtension(RegistrationMode mode) : m_mode(mode) {}

ArchitectureExtension::~ArchitectureExtension() = default;

}  // namespace ov::frontend::gguf
