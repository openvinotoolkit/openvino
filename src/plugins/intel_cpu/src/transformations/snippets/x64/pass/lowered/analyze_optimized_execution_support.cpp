// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "analyze_optimized_execution_support.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

namespace ov::intel_cpu {

bool pass::AnalyzeOptimizedExecutionSupport::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AnalyzeOptimizedExecutionSupport")

    m_is_supported = true;

    // Currently, CPU Plugin doesn't provide optimized execution for Subgraphs with MatMuls `M_dim` = 1
    if (!linear_ir.get_config().m_enable_domain_optimization) {
        for (const auto& expr : linear_ir) {
            if (ov::is_type<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
                const auto planar_dims = ov::snippets::utils::get_planar_vdims(expr->get_input_port(0));
                const auto M_dim = *++planar_dims.rbegin();
                if (ov::snippets::utils::is_dynamic_value(M_dim) || M_dim == 1) {
                    m_is_supported = false;
                    break;
                }
            }
        }
    }

    return m_is_supported;
}
}  // namespace ov::intel_cpu
