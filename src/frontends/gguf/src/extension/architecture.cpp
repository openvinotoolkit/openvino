// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/extension/architecture.hpp"

#include "builder/arch/decoder_builder.hpp"
#include "builder/sdk/metadata_store.hpp"
#include "openvino/core/except.hpp"

namespace ov::frontend::gguf {

bool ArchitectureDefinition::matches(const GgufMetadata& metadata) const {
    return metadata.architecture() == architecture && (!match || match(metadata));
}

ArchitectureDefinition make_decoder_architecture(std::string architecture,
                                                 RopeMode rope,
                                                 DecoderOptionsFn options,
                                                 Maturity maturity) {
    ArchitectureDefinition definition;
    definition.id = definition.architecture = std::move(architecture);
    definition.maturity = maturity;
    definition.factory = [rope, options = std::move(options)](const BuildContext& ctx) {
        OPENVINO_ASSERT(ctx.weights, "[GGUF] decoder builder requires a weight table");
        const auto overrides = options ? options(ctx.metadata) : DecoderOptions{};
        auto config = decoder_config_from_meta(detail::MetadataAccess::get(ctx.metadata).map);
        return std::make_shared<DecoderBuilder>(config, ctx.weights->weights, ctx.weights->qtypes, rope, overrides);
    };
    return definition;
}

ArchitectureExtension::ArchitectureExtension(ArchitectureDefinition definition, RegistrationMode mode)
    : m_definition(std::move(definition)),
      m_mode(mode) {}

ArchitectureExtension::ArchitectureExtension(std::string architecture, RopeMode rope, Maturity maturity)
    : ArchitectureExtension(make_decoder_architecture(std::move(architecture), rope, {}, maturity)) {}

ArchitectureExtension::~ArchitectureExtension() = default;

}  // namespace ov::frontend::gguf
