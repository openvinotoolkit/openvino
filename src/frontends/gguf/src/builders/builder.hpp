// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "gguf_reader.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace frontend {
namespace gguf {

/// \brief Build the qwen3 transformer graph from a parsed GGUF model.
///
/// Weight tensors are emitted as raw gguf_* Constants and consumed by FullyConnectedCompressed
/// nodes (no in-graph dequantization). Throws if a required tensor or metadata key is missing.
std::shared_ptr<ov::Model> build_qwen3_model(const GGUFReader& reader);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
