// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/op/loop.hpp"
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
    InsertLoops() = default;
    bool run(LinearIR& linear_ir) override;
private:
    static void insertion(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager, size_t loop_id);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
