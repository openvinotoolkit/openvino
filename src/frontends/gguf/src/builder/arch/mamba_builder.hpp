// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/frontend/gguf/extension/architecture.hpp"

namespace ov::frontend::gguf {
ArchitectureDefinition mamba2_architecture(const std::string& architecture);
}  // namespace ov::frontend::gguf
