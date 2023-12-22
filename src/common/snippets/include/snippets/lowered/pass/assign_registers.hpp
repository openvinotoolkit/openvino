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
    explicit AssignRegisters(const std::function<Generator::opRegType(const std::shared_ptr<Node>& op)>& mapper) : m_reg_type_mapper(mapper) {}
    bool run(LinearIR& linear_ir) override;

private:
    std::function<Generator::opRegType(const std::shared_ptr<Node>& op)> m_reg_type_mapper;
    static constexpr size_t reg_count = 16lu;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
