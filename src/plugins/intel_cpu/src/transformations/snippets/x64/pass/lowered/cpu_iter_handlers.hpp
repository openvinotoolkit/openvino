// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/iter_handler.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {
class SetBrgemmBeta : public snippets::lowered::pass::SubgraphPass {
public:
    SetBrgemmBeta(float beta);
    OPENVINO_RTTI("SetBrgemmBeta", "Pass")
    bool run(const snippets::lowered::LinearIR& linear_ir,
             snippets::lowered::LinearIR::constExprIt begin,
             snippets::lowered::LinearIR::constExprIt end) override;

private:
    size_t m_beta;
};
}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov