// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/generator.hpp"

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
    OPENVINO_RTTI("AssignRegisters", "Pass")
    explicit AssignRegisters(const std::function<RegType(const ov::Output<Node>& out)>& mapper, const size_t reg_cnt)
                            : m_reg_type_mapper(mapper), reg_count(reg_cnt) {}
    bool run(LinearIR& linear_ir) override;

private:
    void set_reg_types(LinearIR& linear_ir);

    std::function<RegType(const ov::Output<Node>& out)> m_reg_type_mapper;
    size_t reg_count;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
