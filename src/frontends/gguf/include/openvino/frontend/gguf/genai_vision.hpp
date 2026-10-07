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

/// Vision models of a converted mmproj in the optimum-intel export layout that OpenVINO GenAI
/// loads, keyed by GenAI model name: "vision_embeddings" and, for Qwen, "vision_embeddings_pos"
/// and "vision_embeddings_merger". Their inputs are those of the optimum-intel export, so GenAI
/// preprocesses media exactly as for that export. Supports the gemma3, gemma4v, gemma4uv,
/// muse-glimmer and qwen3vl_merger projectors. Where llama.cpp departs from HF, the models follow
/// HF: Gemma4 vision uses tanh GELU. The source model is left unchanged.
GGUF_FRONTEND_API std::map<std::string, std::shared_ptr<ov::Model>> genai_vision_models(
    const std::shared_ptr<ov::Model>& mmproj);

}  // namespace ov::frontend::gguf
