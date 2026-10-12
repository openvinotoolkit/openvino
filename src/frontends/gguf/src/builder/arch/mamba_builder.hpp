// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/frontend/gguf/builder/model_builder.hpp"

namespace ov::frontend::gguf {
std::shared_ptr<ModelBuilder> make_mamba2_builder(const BuildContext& context);
}  // namespace ov::frontend::gguf
