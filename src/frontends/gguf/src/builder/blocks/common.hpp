// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "builder/graph_emitter.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace blocks {

// Small, architecture-agnostic graph fragments shared by every model family.
//
// They are free functions over GraphEmitter rather than methods on a builder so a future
// non-decoder family (mmproj/vision, audio) can reuse them without inheriting any LLM
// hyperparameters. Anything that needs to know about heads, layers or KV caches belongs in a
// decoder-specific block instead (see ffn.hpp / attention.hpp).

// RMS_NORM followed by elementwise MUL with the norm weight (llama.cpp build_norm, LLM_NORM_RMS).
// `eps` is the epsilon fed to the RMS_NORM op.
std::string rms_norm(GraphEmitter& e,
                     const std::string& in,
                     const std::string& weight,
                     const std::string& out_prefix,
                     float eps);

// Scale a tensor by a constant: GGML_OP_SCALE with attr "scale" (and bias 0).
std::string scale(GraphEmitter& e, const std::string& x, float factor, const std::string& name);

// Elementwise add of a (broadcast) bias weight: GGML_OP_ADD(x, bias_weight).
std::string add_bias(GraphEmitter& e, const std::string& x, const std::string& bias_weight, const std::string& name);

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
