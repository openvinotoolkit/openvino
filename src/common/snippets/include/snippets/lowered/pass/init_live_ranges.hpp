// Copyright (C) 2025 Intel Corporation
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
 * @interface InitLiveRanges
 * @brief Calculates live ranges of registers. This information will be used to assign registers and optimize ABI reg spills.
 * @ingroup snippets
 */
class InitLiveRanges : public Pass {
public:
    OPENVINO_RTTI("InitLiveRanges", "", Pass)
    explicit InitLiveRanges(RegManager& reg_manager) : m_reg_manager(reg_manager) {}
    bool run(LinearIR& linear_ir) override;
private:
    RegManager& m_reg_manager;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
