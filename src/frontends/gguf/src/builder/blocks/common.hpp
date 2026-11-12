// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

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

// NORM -> MUL(weight) -> ADD(bias).
std::string layer_norm(GraphEmitter& e,
                       const std::string& in,
                       const std::string& weight,
                       const std::string& out_prefix,
                       float eps);

// Scale a tensor by a constant: GGML_OP_SCALE with attr "scale" (and bias 0).
std::string scale(GraphEmitter& e, const std::string& x, float factor, const std::string& name);

// GGML_OP_RESHAPE with special_zero: a 0 in `target` keeps the input dim, -1 is inferred.
std::string reshape(GraphEmitter& e, const std::string& x, std::vector<int64_t> target, const std::string& name);

// "<name>.weight" -> "<name>.bias".
std::string bias_weight_name(const std::string& weight_name);

// Elementwise add of a (broadcast) bias weight: GGML_OP_ADD(x, bias_weight).
std::string add_bias(GraphEmitter& e, const std::string& x, const std::string& bias_weight, const std::string& name);

std::array<std::string, 2> split_interleaved_projection(GraphEmitter& e,
                                                        const std::string& input,
                                                        const std::array<std::string, 2>& names,
                                                        int64_t groups,
                                                        int64_t width);

std::string concat_weight_rows(GraphEmitter& e, const std::string& name, const std::vector<std::string>& weights);

}  // namespace ov::frontend::gguf::blocks
