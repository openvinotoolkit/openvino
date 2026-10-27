// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Load-time GGUF dequantization for graph paths that cannot consume raw block weights.
//
// GGUF FullyConnected weights stay compressed when the target device reports native support for the
// corresponding block type. Unsupported GGUF tensors, and embedding tensors used by Gather, are
// materialized to dense f16 Constants here at load time.

#pragma once

#include <memory>
#include <string>

#include "gguf_reader.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace gguf {

/// \brief Dequantize the named GGUF tensor to a dense f16 Constant.
///
/// Trivial pass-through for F32/F16/BF16; full block decode for supported GGUF layouts. Throws for
/// unsupported quantization types.
std::shared_ptr<ov::op::v0::Constant> dequantize_to_f16(const GGUFReader& reader, const std::string& name);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
