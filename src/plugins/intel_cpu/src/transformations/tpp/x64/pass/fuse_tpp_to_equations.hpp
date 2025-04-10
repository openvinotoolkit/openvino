// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::tpp::pass {

/**
 * @interface FuseToEquationsTPP
 * @brief Converts a group of elementwise operations into a fused TPP Equation node
 * @ingroup snippets
 */
class FuseTPPToEquations : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FuseTPPToEquations");
    FuseTPPToEquations() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    static bool fuse_from_root(const std::shared_ptr<ov::Node>&, const std::shared_ptr<ov::Model>& m);
};

}  // namespace ov::intel_cpu::tpp::pass
