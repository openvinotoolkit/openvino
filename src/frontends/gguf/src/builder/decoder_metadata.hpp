// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/extension/architecture.hpp"
#include "quant/gguf.hpp"

namespace ov::frontend::gguf {

DecoderParameters decoder_family_defaults(const std::string& architecture, int layers, int embedding, int heads);
DecoderParameters resolve_decoder_parameters(const GgufMetadata& metadata,
                                             RopeMode rope,
                                             const DecoderOptions& options = {});
void apply_decoder_options(DecoderParameters& parameters, const DecoderOptions& options);
DecoderParameters decoder_parameters_from_config(const std::map<std::string, GGUFMetaData>& config);
std::map<std::string, GGUFMetaData> decoder_config_from_parameters(const DecoderParameters& parameters);
std::map<std::string, GGUFMetaData> decoder_weight_transform_metadata(const GgufMetadata& metadata,
                                                                      int value_heads,
                                                                      int key_heads);
std::map<std::string, GGUFMetaData> decoder_config_from_meta(
    const std::unordered_map<std::string, GGUFMetaData>& metadata);
ArchitectureDefinition::BuilderFactory decoder_builder_factory(RopeMode rope, DecoderOptionsFn options);
ArchitectureDefinition::BuilderFactory decoder_builder_factory(DecoderParametersFn parameters);

}  // namespace ov::frontend::gguf
