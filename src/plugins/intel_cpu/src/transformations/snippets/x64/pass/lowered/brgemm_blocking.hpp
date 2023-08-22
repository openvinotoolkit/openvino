// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov {
namespace intel_cpu {
namespace pass {

/**
 * @interface BrgemmBlocking
 * @brief Covers BrgemmCPU with blocking loop by M
 * @ingroup snippets
 */

class BrgemmBlocking : public snippets::lowered::pass::Pass {
public:
    OPENVINO_RTTI("BrgemmBlocking", "Pass")
    BrgemmBlocking();
    bool run(snippets::lowered::LinearIR& linear_ir) override;

private:
    static void move_new_memory_buffer(snippets::lowered::LinearIR& linear_ir, const snippets::lowered::LinearIR::constExprIt& brgemm_it);
};

}  // namespace pass
}  // namespace intel_cpu
}  // namespace ov