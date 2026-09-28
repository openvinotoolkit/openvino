// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @brief Base pass for adjusting loop ports after CopyB extraction.
 */
class AdjustCopyBLoopPorts : public snippets::lowered::pass::ConstPass {
public:
    bool run(const snippets::lowered::LinearIR& linear_ir) override;

    const std::unordered_set<snippets::lowered::UnifiedLoopInfoPtr>& get_affected_loops() const {
        return m_affected_loops;
    }

protected:
    static void assign_new_ptr_increment(int64_t new_ptr_increment,
                                         snippets::lowered::UnifiedLoopInfo::LoopPortDesc& loop_desc);
    /// Returns true when @p expr is a backend MatMul operation that needs CopyB loop port adjustment.
    virtual bool is_target_expr(const snippets::lowered::ExpressionPtr& expr) const = 0;
    /// Returns the CopyB expression that supplies the weights for @p gemm_expr.
    virtual snippets::lowered::ExpressionPtr get_copy_b_expr(
        const snippets::lowered::ExpressionPtr& gemm_expr) const = 0;
    /// Adjusts all affected ports in @p loop_info for the backend layout.
    virtual bool update_loop_info_impl(const snippets::lowered::UnifiedLoopInfoPtr& loop_info) const = 0;
    /// Returns the diagnostic used when the expected CopyB expression is absent.
    virtual const char* copy_b_not_found_message() const = 0;
    /// Returns the diagnostic used for invalid MatMul and CopyB loop nesting.
    virtual const char* invalid_loop_config_message() const = 0;

private:
    std::unordered_set<snippets::lowered::UnifiedLoopInfoPtr> m_affected_loops;
};

}  // namespace ov::intel_cpu::pass
