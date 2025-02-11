// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/generator.hpp"
#include "snippets/lowered/reg_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface AssignRegisters
 * @brief Assigns in/out abstract registers indexes to every operation.
 * Note that changing of the IR is likely to invalidate register assignment.
 * @ingroup snippets
 */
class AssignRegisters : public Pass {
public:
    OPENVINO_RTTI("AssignRegisters", "0", Pass)
    explicit AssignRegisters(RegManager& reg_manager) : m_reg_manager(reg_manager) {}
    bool run(LinearIR& linear_ir) override;

private:
    using RegMap = std::map<Reg, Reg>;
    RegMap assign_regs_manually(const LinearIR& linear_ir, std::set<Reg>& gpr_pool, std::set<Reg>& vec_pool);

    RegManager& m_reg_manager;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
