// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kv_axes_position.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

class ReshapeToStatic : public ov::pass::ModelPass {
    uint32_t m_input_size;
    uint32_t m_kvcache_size;
    KVAxesPosition m_kv_axes_position;
    uint32_t m_lora_rank;
    uint32_t m_lhs_seq_size;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ReshapeToStatic");
    explicit ReshapeToStatic(const uint32_t input_size,
                             const uint32_t kvcache_size,
                             const KVAxesPosition& kv_axes_position,
                             const uint32_t lora_rank,
                             const uint32_t lhs_seq_size = 0);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
