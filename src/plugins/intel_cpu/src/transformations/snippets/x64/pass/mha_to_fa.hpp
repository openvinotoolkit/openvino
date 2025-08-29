// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface MHAToFA
 * @brief Replaces MHA with snippets::op::Flash_Attention operation
 * @ingroup snippets
 */
class MHAToFA : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MHAToFA");
    MHAToFA() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu::pass
