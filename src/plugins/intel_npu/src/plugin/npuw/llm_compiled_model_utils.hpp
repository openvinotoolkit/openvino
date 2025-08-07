// Copyright (C) 2024-2025 Intel Corporation
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

// SDPA-unroll and transpose transformations
bool optimize_value_tensors(std::shared_ptr<ov::Model> model, bool isPrefill);


std::shared_ptr<ov::Model> prepare_whisper_prefill_model(std::shared_ptr<ov::Model>& model,
                                                         const uint32_t& max_prompt_size,
                                                         const uint32_t& lhs_seq_size);

std::shared_ptr<ov::Model> prepare_whisper_kvcache_model(std::shared_ptr<ov::Model>& model);

// clang-format off
}  // namespace ov
// clang-format on
