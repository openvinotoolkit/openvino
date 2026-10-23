// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/builder/model_builder.hpp"

namespace ov::frontend::gguf {
std::shared_ptr<ModelBuilder> make_mmproj_builder(const BuildContext& context);
std::shared_ptr<ModelBuilder> make_qwen3tts_projector_builder(const BuildContext& context);
}  // namespace ov::frontend::gguf
