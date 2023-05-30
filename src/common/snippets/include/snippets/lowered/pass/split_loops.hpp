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
 * @interface SplitLoops
 * @brief If a loop has larger increment than another but similar works amount, then split an outer loop into two
 *        so the outermost of the two could be fused with the second loop.
 * @ingroup snippets
 */

class SplitLoops : public Pass {
public:
    OPENVINO_RTTI("SplitLoops", "Pass")
    SplitLoops();
    bool run(LinearIR& linear_ir) override;

private:
    static bool can_be_split(const LinearIR::LoopManager::LoopInfoPtr& current,
                             const LinearIR::LoopManager::LoopInfoPtr& target);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov