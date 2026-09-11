// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "transformations/snippets/common/pass/lowered/adjust_copy_b_loop_ports.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface AdjustBrgemmCopyBLoopPorts
 * @brief BrgemmCopyB is located outside of blocking loops and repacks input data into a blocked layout.
 *        This layout should be accounted for when we increment BrgemmCopyB data pointers. This pass
 *        Finds loop ports connected to BrgemmCopyB and sets appropriate pointer increments.
 * @ingroup snippets
 */
class AdjustBrgemmCopyBLoopPorts : public AdjustCopyBLoopPorts {
public:
    OPENVINO_RTTI("AdjustBrgemmCopyBLoopPorts", "", AdjustCopyBLoopPorts)
    AdjustBrgemmCopyBLoopPorts() = default;
    static bool update_loop_info(const snippets::lowered::UnifiedLoopInfoPtr& uni_loop_info);

private:
    bool is_target_expr(const snippets::lowered::ExpressionPtr& expr) const override;
    snippets::lowered::ExpressionPtr get_copy_b_expr(const snippets::lowered::ExpressionPtr& gemm_expr) const override;
    bool update_loop_info_impl(const snippets::lowered::UnifiedLoopInfoPtr& loop_info) const override;
    const char* copy_b_not_found_message() const override;
    const char* invalid_loop_config_message() const override;
};

}  // namespace ov::intel_cpu::pass
