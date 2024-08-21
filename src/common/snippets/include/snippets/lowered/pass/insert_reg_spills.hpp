// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/lowered/reg_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertRegSpills
 * @brief Insert RegSpill and RegRestore operations for binary call emitters to comply with ABI conventions.
 * @ingroup snippets
 */
class InsertRegSpills : public Pass {
public:
    OPENVINO_RTTI("InsertRegSpills", "Pass")
    explicit InsertRegSpills(RegManager& reg_manager) : m_reg_manager(reg_manager) {}
    bool run(LinearIR& linear_ir) override;

    RegManager& m_reg_manager;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
