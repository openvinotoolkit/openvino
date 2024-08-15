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
 * @brief The pass initializes scheduling information in UnifiedLoopInfo
 * @ingroup snippets
 */
class InitLoops : public Pass {
public:
    OPENVINO_RTTI("InitLoops", "Pass")
    InitLoops() = default;
    bool run(LinearIR& linear_ir) override;

    /**
     * @brief Updates ptr_increments and finalization offsets of the provided "loop_info" based on current work amount
     */
    static void update_data_pointer_shifts(const UnifiedLoopInfoPtr& loop_info);
    /**
     * @brief Updates work amount and updates data pointer shifts of the provided "loop_info"
     */
    static void update_runtime_parameters(const UnifiedLoopInfoPtr& loop_info);

private:
    static void update_compile_parameters(const UnifiedLoopInfoPtr& loop_info, size_t loop_id);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
