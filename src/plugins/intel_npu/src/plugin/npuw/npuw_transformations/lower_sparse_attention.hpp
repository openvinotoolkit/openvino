// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

class LowerSparseAttention : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::LowerSparseAttention");
    explicit LowerSparseAttention(std::shared_ptr<ov::Node> physical_attention_mask = nullptr)
        : m_physical_attention_mask(std::move(physical_attention_mask)) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<ov::Node> m_physical_attention_mask;
};

class LowerSparseAttentionToGathered : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::LowerSparseAttentionToGathered");
    explicit LowerSparseAttentionToGathered(std::shared_ptr<ov::Node> physical_attention_mask)
        : m_physical_attention_mask(std::move(physical_attention_mask)) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<ov::Node> m_physical_attention_mask;
};

}  // namespace ov::npuw
