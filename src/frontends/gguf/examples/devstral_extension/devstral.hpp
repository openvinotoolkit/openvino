// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/extension/architecture.hpp"

namespace example {
inline ov::frontend::gguf::ArchitectureDefinition devstral_decoder(const std::string& architecture) {
    using namespace ov::frontend::gguf;
    // Devstral uses existing GGUF family names; its dimensions and scaling come from metadata.
    return make_decoder_architecture(architecture, RopeMode::Normal);
}
}  // namespace example
