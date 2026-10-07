// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <set>
#include <vector>

#include "openvino/frontend/gguf/extension/architecture.hpp"

namespace ov::frontend::gguf {

constexpr int ROPE_OP_CASE_NORMAL = 0x00000000;
constexpr int ROPE_OP_CASE_NEOX = 0x00010000;
constexpr int ROPE_OP_CASE_IMROPE = 0x00020000;
bool arch_uses_neox_rope(const std::string& arch);
const std::set<std::string>& verified_archs();
const std::set<std::string>& experimental_archs();
const std::set<std::string>& supported_archs();

// Single catalog for built-in decoder and custom-family definitions.
std::vector<ArchitectureDefinition> builtin_architectures();

class ArchRegistry {
public:
    explicit ArchRegistry(std::vector<ArchitectureDefinition> definitions = builtin_architectures());
    void add(ArchitectureDefinition definition, RegistrationMode mode = RegistrationMode::Add);
    void add_extension(const ArchitectureExtension::Ptr& ext);
    std::shared_ptr<const ArchitectureDefinition> find(const GgufMetadata& meta) const;
    std::string describe_supported() const;

private:
    std::map<std::string, std::shared_ptr<const ArchitectureDefinition>> m_definitions;
};

const ArchRegistry& default_arch_registry();

}  // namespace ov::frontend::gguf
