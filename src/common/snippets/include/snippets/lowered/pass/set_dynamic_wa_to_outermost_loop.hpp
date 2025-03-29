// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SetDynamicWAToOuterMostLoop
 * @brief The pass set dynamic work amount to outermost Loop by M in dynamic MHA Subgraphs
 *        to allow MHAParallelWAOptimizer optimizes parallel work amount in runtime.
 * @ingroup snippets
 */
class SetDynamicWAToOuterMostLoop : public Pass {
public:
    OPENVINO_RTTI("SetDynamicWAToOuterMostLoop", "", Pass);
    SetDynamicWAToOuterMostLoop() = default;
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
