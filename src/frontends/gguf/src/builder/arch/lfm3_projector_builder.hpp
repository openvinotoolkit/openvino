// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/extension/projector.hpp"

namespace ov::frontend::gguf {
ProjectorResult build_lfm3_projector(GgufGraphContext& graph, const std::string& modality);
std::vector<ProjectorDefinition> lfm3_projector_definitions();
}  // namespace ov::frontend::gguf
