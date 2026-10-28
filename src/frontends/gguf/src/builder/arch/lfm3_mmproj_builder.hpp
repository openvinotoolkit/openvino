// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/builder/model_builder.hpp"

namespace ov::frontend::gguf {
bool is_lfm3_mmproj(const GgufMetadata& metadata);
std::shared_ptr<ModelBuilder> make_lfm3_mmproj_builder(const BuildContext& context);
}  // namespace ov::frontend::gguf
