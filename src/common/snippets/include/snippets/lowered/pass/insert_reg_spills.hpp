// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <utility>

#include "openvino/core/rtti.hpp"
#include "openvino/core/type.hpp"
#include "pass.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/reg_manager.hpp"
#include "snippets/op/brgemm.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @brief Default function to enable RegSpill insertion
 * @return True if RegSpill is required around a certain op, False otherwise.
 */
inline bool needs_reg_spill_default(const ExpressionPtr& expr) {
    return ov::is_type<snippets::op::Brgemm>(expr->get_node());
}

/**
 * @interface InsertRegSpills
 * @brief Insert RegSpill and RegRestore operations for binary call emitters to comply with ABI conventions.
 * @ingroup snippets
 */
class InsertRegSpills : public Pass {
public:
    OPENVINO_RTTI("InsertRegSpills", "", Pass)
    explicit InsertRegSpills(RegManager& reg_manager,
                             std::function<bool(const ExpressionPtr&)> needs_reg_spill = needs_reg_spill_default)
        : m_reg_manager(reg_manager),
          m_needs_reg_spill(std::move(needs_reg_spill)) {}
    bool run(LinearIR& linear_ir) override;

private:
    RegManager& m_reg_manager;
    std::function<bool(const ExpressionPtr&)> m_needs_reg_spill;
};

}  // namespace ov::snippets::lowered::pass
