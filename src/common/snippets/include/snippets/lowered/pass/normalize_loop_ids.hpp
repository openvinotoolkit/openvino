// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface NormalizeLoopIDs
 * @brief After Loop optimizations loop may have unsorted and unevenly IDs (some IDs may be missed).
 *        The pass normalize IDs of Loops in Linear IR and update LoopManager.
 * @ingroup snippets
 */
class NormalizeLoopIDs : public Pass {
public:
    OPENVINO_RTTI("NormalizeLoopIDs", "Pass")
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
