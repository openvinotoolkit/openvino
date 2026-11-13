// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/extension/architecture.hpp"
#include "openvino/frontend/gguf/extension/projector.hpp"

namespace example {
ov::frontend::gguf::ArchitectureDefinition projector_architecture();
ov::frontend::gguf::ProjectorDefinition vision_projector_component();
}  // namespace example
