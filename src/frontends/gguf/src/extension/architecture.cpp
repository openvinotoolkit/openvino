// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/extension/architecture.hpp"

#include <utility>

namespace ov {
namespace frontend {
namespace gguf {

ArchitectureExtension::ArchitectureExtension(std::string architecture, RopeMode rope, Maturity maturity)
    : m_architecture(std::move(architecture)),
      m_rope(rope),
      m_maturity(maturity) {}

ArchitectureExtension::ArchitectureExtension(std::string architecture,
                                             RopeMode rope,
                                             ConfigureFn configure,
                                             Maturity maturity)
    : m_architecture(std::move(architecture)),
      m_rope(rope),
      m_maturity(maturity),
      m_configure(std::move(configure)) {}

ArchitectureExtension::ArchitectureExtension(std::string architecture,
                                             BuilderFactory factory,
                                             MatchFn match,
                                             Maturity maturity)
    : m_architecture(std::move(architecture)),
      m_maturity(maturity),
      m_factory(std::move(factory)),
      m_match(std::move(match)) {}

ArchitectureExtension::~ArchitectureExtension() = default;

bool ArchitectureExtension::matches(const GgufMetadata& meta) const {
    // A predicate takes precedence: it exists precisely for a file whose `general.architecture`
    // does not identify it (an mmproj file calls itself "clip" whatever it actually holds).
    if (m_match) {
        return m_match(meta);
    }
    return meta.architecture() == m_architecture;
}

void ArchitectureExtension::configure(DecoderConfig& config) const {
    if (m_configure) {
        m_configure(config);
    }
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
