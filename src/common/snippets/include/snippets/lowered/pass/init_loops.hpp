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
 * @brief The pass initializes scheduling information in LoopInfo
 * @ingroup snippets
 */
class InitLoops : public Pass {
public:
    OPENVINO_RTTI("InitLoops", "Pass")
    InitLoops();
    bool run(LinearIR& linear_ir) override;

private:
    static void init_ptr_increments(const LinearIR::LoopManager::LoopInfoPtr& loop_info);
    static void init_finalization_offsets(const LinearIR::LoopManager::LoopInfoPtr& loop_info);
    static void init_element_type_sizes(const LinearIR::LoopManager::LoopInfoPtr& loop_info);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
