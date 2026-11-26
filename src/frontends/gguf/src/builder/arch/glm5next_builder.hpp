// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/frontend/gguf/builder/model_builder.hpp"

namespace ov::frontend::gguf {

// GLM5-Next (GLM-5.3-Flash): mHC residual streams around Kimi Delta Attention layers and nope MLA layers with a
// k-pool DSA indexer, DeepSeek-style sigmoid MoE. Both architecture spellings seen in GGUF files, "glm5next" and
// "glm5-next", are accepted.
std::shared_ptr<ModelBuilder> make_glm5next_builder(const BuildContext& context);

}  // namespace ov::frontend::gguf
