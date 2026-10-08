// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov::frontend::gguf {

/// Vision models with inputs compatible with the shared OpenVINO GenAI encoders, keyed by
/// "vision_embeddings" and, for Qwen, "vision_embeddings_pos" and "vision_embeddings_merger".
/// Supports gemma3, gemma4v, gemma4uv, muse-glimmer and qwen3vl_merger. The adapted models
/// preserve llama.cpp computation, including GGUF activation choices. GenAI must configure
/// preprocessing from GGUF geometry and llama.cpp limits. The source model is left unchanged.
GGUF_FRONTEND_API std::map<std::string, std::shared_ptr<ov::Model>> genai_vision_models(
    const std::shared_ptr<ov::Model>& mmproj);

}  // namespace ov::frontend::gguf
