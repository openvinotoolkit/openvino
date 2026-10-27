// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// qwen35 requires linear_attn / gated_delta_net ops not present in this build.
// Stub: builder.hpp declares the function; this TU provides the (unreachable) definition.
#include "builders/builder.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace frontend {
namespace gguf {

std::shared_ptr<ov::Model> build_qwen35_model(const GGUFReader&) {
    OPENVINO_THROW("[GGUF Frontend] qwen35 architecture is not supported in this build "
                  "(requires linear_attn / gated_delta_net ops).");
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
