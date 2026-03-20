// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

class SliceOutEmbeds : public ov::pass::ModelPass {
    uint32_t m_batch_dim;
    std::size_t m_max_generation_token_len;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::SliceOutEmbeds");
    explicit SliceOutEmbeds(uint32_t batch_dim, std::size_t max_generation_token_len);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
