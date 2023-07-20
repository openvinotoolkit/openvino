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
 * @interface InitLoops
 * @brief The pass initialize scheduling information in LoopInfo
 * @ingroup snippets
 */
class InitLoops : public Pass {
public:
    OPENVINO_RTTI("InitLoops", "Pass")
    InitLoops();
    bool run(LinearIR& linear_ir) override;

private:
    static void init_ptr_increments(std::vector<LinearIR::LoopManager::LoopPort>& loop_inputs,
                                    std::vector<LinearIR::LoopManager::LoopPort>& loop_outputs,
                                    size_t work_amount, size_t dim_idx);
    static void init_finalization_offsets(std::vector<LinearIR::LoopManager::LoopPort>& loop_inputs,
                                          std::vector<LinearIR::LoopManager::LoopPort>& loop_outputs,
                                          size_t work_amount);
    static void init_element_type_sizes(std::vector<LinearIR::LoopManager::LoopPort>& loop_inputs,
                                        std::vector<LinearIR::LoopManager::LoopPort>& loop_outputs);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
