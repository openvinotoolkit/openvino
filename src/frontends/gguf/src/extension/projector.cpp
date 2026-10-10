// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/extension/projector.hpp"

namespace ov::frontend::gguf {

ProjectorExtension::ProjectorExtension(ProjectorDefinition definition, RegistrationMode mode)
    : ArchitectureExtension(mode),
      m_projector(std::move(definition)) {}

ProjectorExtension::~ProjectorExtension() = default;

}  // namespace ov::frontend::gguf
