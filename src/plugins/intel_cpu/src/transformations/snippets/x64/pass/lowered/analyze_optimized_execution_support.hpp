// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface AnalyzeOptimizedExecutionSupport
 * @brief Check if the current LinearIR configuration can be efficiently executed by CPU.
 * @ingroup snippets
 */
class AnalyzeOptimizedExecutionSupport : public ov::snippets::lowered::pass::ConstPass {
public:
    OPENVINO_RTTI("AnalyzeOptimizedExecutionSupport", "", ov::snippets::lowered::pass::ConstPass)
    AnalyzeOptimizedExecutionSupport(bool& is_supported) : m_is_supported(is_supported) {}

    bool run(const ov::snippets::lowered::LinearIR& linear_ir) override;

private:
    bool& m_is_supported;
};

}  // namespace ov::intel_cpu::pass
