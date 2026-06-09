// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Embedding-only dequantization.
//
// Per the GGUF frontend contract, FullyConnected weights are kept compressed (raw gguf_* Constants
// consumed by FullyConnectedCompressed). The token embedding is a Gather (row lookup), not a GEMM,
// and there is no gguf-aware Gather kernel; therefore the single tensor backing the embedding is
// materialized to a dense f16 Constant here, at load time. No gguf type ever reaches the graph for
// this path, so the no-in-graph-dequantization invariant is preserved.

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
/// Trivial pass-through for F32/F16/BF16; full block decode for Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K
/// (canonical ggml layouts). Throws for unsupported quantization types.
std::shared_ptr<ov::op::v0::Constant> dequantize_to_f16(const GGUFReader& reader, const std::string& name);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
