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

// SDPA-unroll and transpose transformations
class OptimizeValueTensors : public ov::pass::ModelPass {
    bool m_is_prefill;
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::OptimizeValueTensors");
    explicit OptimizeValueTensors(bool is_prefill) : m_is_prefill(is_prefill) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

class PrepareWhisperPrefillModel : public ov::pass::ModelPass {
    uint32_t m_max_prompt_size;
    uint32_t m_lhs_seq_size;
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PrepareWhisperPrefillModel");
    explicit PrepareWhisperPrefillModel(uint32_t max_prompt_size, uint32_t lhs_seq_size)
        : m_max_prompt_size(max_prompt_size), m_lhs_seq_size(lhs_seq_size) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

class PrepareWhisperKVCacheModel : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PrepareWhisperKVCacheModel");
    PrepareWhisperKVCacheModel() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// clang-format off
}  // namespace ov
// clang-format on
