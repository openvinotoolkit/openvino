// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"

namespace ov::snippets::lowered::pass {

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

}  // namespace ov::snippets::lowered::pass
