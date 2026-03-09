// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace util {

class PrepareWhisperPrefillModel : public ov::pass::ModelPass {
    uint32_t m_max_prompt_size;
    uint32_t m_lhs_seq_size;
    bool m_decompose_sdpa;
    uint32_t m_decomposed_layers_size;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PrepareWhisperPrefillModel");
    explicit PrepareWhisperPrefillModel(uint32_t max_prompt_size, uint32_t lhs_seq_size, bool decompose_sdpa)
        : m_max_prompt_size(max_prompt_size),
          m_lhs_seq_size(lhs_seq_size),
          m_decompose_sdpa(decompose_sdpa),
          m_decomposed_layers_size(0) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

class PrepareWhisperKVCacheModel : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PrepareWhisperKVCacheModel");
    PrepareWhisperKVCacheModel() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace util
}  // namespace npuw
}  // namespace ov
