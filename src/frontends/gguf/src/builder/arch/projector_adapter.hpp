// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/frontend/gguf/extension/projector.hpp"

namespace ov::frontend::gguf {

using ProjectorModels = std::map<std::string, std::shared_ptr<ov::Model>>;

std::string projector_layout_name(ProjectorLayout layout);
// Vision GenAI layout of a built-in projector type; Unsupported for other types.
ProjectorLayout builtin_projector_layout(const std::string& projector_type);
void require_projector_layout(const std::shared_ptr<ov::Model>& model);
ProjectorModels adapt_projector_layout(const std::shared_ptr<ov::Model>& model);

}  // namespace ov::frontend::gguf
