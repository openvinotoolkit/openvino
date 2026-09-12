// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "openvino/core/extension.hpp"
#include "openvino/frontend/gguf/builder/decoder_options.hpp"
#include "openvino/frontend/gguf/builder/model_builder.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov::frontend::gguf {

enum class Maturity { Experimental, Verified };
enum class RegistrationMode { Add, Replace };

// Shared by built-in and external registration; promotion preserves the factory and builder.
struct GGUF_FRONTEND_API ArchitectureDefinition {
    using BuilderFactory = std::function<std::shared_ptr<ModelBuilder>(const BuildContext&)>;
    using MatchFn = std::function<bool(const GgufMetadata&)>;

    std::string id;            // Unique handler identity, e.g. "clip.vision".
    std::string architecture;  // Required general.architecture, e.g. "clip".
    BuilderFactory factory;
    MatchFn match;  // Optional additional constraint within this architecture.
    Maturity maturity = Maturity::Experimental;

    bool matches(const GgufMetadata& metadata) const;
};

// Use the native decoder resolver and blocks. Options are collected before configuration is resolved.
using DecoderOptionsFn = std::function<DecoderOptions(const GgufMetadata&)>;
GGUF_FRONTEND_API ArchitectureDefinition make_decoder_architecture(std::string architecture,
                                                                   RopeMode rope,
                                                                   DecoderOptionsFn options = {},
                                                                   Maturity maturity = Maturity::Experimental);

class GGUF_FRONTEND_API ArchitectureExtension : public ov::Extension {
public:
    OPENVINO_RTTI("gguf::ArchitectureExtension", "", ov::Extension);
    using Ptr = std::shared_ptr<ArchitectureExtension>;

    explicit ArchitectureExtension(ArchitectureDefinition definition, RegistrationMode mode = RegistrationMode::Add);
    ArchitectureExtension(std::string architecture, RopeMode rope, Maturity maturity = Maturity::Experimental);
    ~ArchitectureExtension() override;

    const ArchitectureDefinition& definition() const {
        return m_definition;
    }
    RegistrationMode registration_mode() const {
        return m_mode;
    }

private:
    ArchitectureDefinition m_definition;
    RegistrationMode m_mode;
};

}  // namespace ov::frontend::gguf
