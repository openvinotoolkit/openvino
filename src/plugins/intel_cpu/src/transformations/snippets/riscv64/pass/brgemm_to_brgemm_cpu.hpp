// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface BrgemmToBrgemmCPU
 * @brief Replaces a Snippets Brgemm with the RV64 BrgemmCPU operation.
 * @ingroup snippets
 */
class BrgemmToBrgemmCPU : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("BrgemmToBrgemmCPU");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu::pass
