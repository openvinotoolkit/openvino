// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>

#include "openvino/openvino.hpp"

namespace ov ::npuw ::util {

/*
 * special mark on nodes to be remain in high-precision for optimal processing
 */
class HighPrecisionAttr : public RuntimeAttribute {
public:
    OPENVINO_RTTI("HighPrecisionAttr", "0", RuntimeAttribute);
    ov::element::Type compute_precision_type;

    bool visit_attributes(AttributeVisitor& visitor) override {
        visitor.on_attribute("compute_precision", compute_precision_type);
        return true;
    }
};

bool has_input(const std::shared_ptr<ov::Model>& model, const std::string& name);

// Returns true for a non-autoregressive (bidirectional encoder, e.g. BERT) text-embedding
// model: one that has ScaledDotProductAttention but NO autoregressive KV-cache concat pattern
// (Concat->Broadcast->Reshape on the SDPA key input) that the Qwen3-Embedding-style path needs.
// Used to route encoder embedders to the dedicated, KV/RoPE-free embedding path.
bool is_encoder_embedding_model(const std::shared_ptr<ov::Model>& model);

// clang-format off
}  // namespace ov
// clang-format on
