// Copyright (C) 2023-2024 Intel Corporation
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
    OPENVINO_RTTI("InitLoops", "", Pass);
    InitLoops() = default;
    bool run(LinearIR& linear_ir) override;

private:
    static void update_compile_parameters(const UnifiedLoopInfoPtr& loop_info);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
