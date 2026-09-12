// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/extension/architecture.hpp"

namespace example {
ov::frontend::gguf::ArchitectureDefinition devstral_decoder(const std::string& architecture);
}  // namespace example
