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
    InitLoops() = default;
    bool run(LinearIR& linear_ir) override;

    static void init_loop_info(const LoopInfoPtr& loop_info, bool only_runtime_args = false);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
