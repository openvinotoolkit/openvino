// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

class SelectLastLogits : public ov::pass::ModelPass {
    uint32_t m_batch_dim;
    std::size_t m_max_generation_token_len;
    std::string m_output_name;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::SelectLastLogits");
    SelectLastLogits(uint32_t batch_dim, std::size_t max_generation_token_len, std::string output_name);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
