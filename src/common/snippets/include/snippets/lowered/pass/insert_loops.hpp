// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertLoops
 * @brief The pass explicitly insert LoadBegin and LoadEnd in Linear IR using LoopManager::LoopInfo from Loop markup algorithm
 * @ingroup snippets
 */
class InsertLoops : public Pass {
public:
    OPENVINO_RTTI("InsertLoops", "Pass")
    InsertLoops();
    bool run(LinearIR& linear_ir) override;
private:
    static void insertion(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager, size_t loop_id, bool has_outer_loop);
    static void filter_ports(std::vector<LinearIR::LoopManager::LoopPort>& loop_entries, std::vector<LinearIR::LoopManager::LoopPort>& loop_exits);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
