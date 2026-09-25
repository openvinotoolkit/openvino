// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string>
#include <utility>

#include "builder/graph_emitter.hpp"

namespace ov::frontend::gguf::blocks {

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

// The extracted weight/scales/zero-point tensors of weight `base` ("<base>.weight" etc.).
WeightTensors weight_parts(GraphEmitter& e, const std::string& base);

// Quant type of weight `base`; F16 when unknown.
GgufTensorType weight_qtype(GraphEmitter& e, const std::string& base);

// Register the tensors of a derived weight `base` with quant type `qtype`.
void store_parts(GraphEmitter& e, const std::string& base, const WeightTensors& t, GgufTensorType qtype);

// Concatenate the rows of two weights with the same quantization layout into one; empty on mismatch.
std::optional<WeightTensors> concat_rows(const WeightTensors& a,
                                         const WeightTensors& b,
                                         GgufTensorType qa,
                                         GgufTensorType qb);

// Like concat_rows, but when the layouts differ, first re-express both 32-group 4/8-bit weights
// exactly in a common 8-bit layout (Q8_0: i8 symmetric, or Q5_K: u8 + u8 zero-point). Returns the
// merged tensors and the quant type describing their layout; empty when not representable.
std::optional<std::pair<WeightTensors, GgufTensorType>> concat_rows_widened(const WeightTensors& a,
                                                                            const WeightTensors& b,
                                                                            GgufTensorType qa,
                                                                            GgufTensorType qb);

}  // namespace ov::frontend::gguf::blocks
