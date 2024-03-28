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
 * @brief Sort loop IDs in LoopEnd expressions order and make them dense with expression loop IDs updates.
 *        After optimizations on Loops IDs might be unevenly: some numbers are missed and unsorted.
 *        For example,
 *             [Loop -> ID -> new ID]
 *              Loop0 -> 3 -> 0
 *              Loop1 -> 0 -> 1
 *              Loop2 -> 1 -> 2
 *        Note: If the LinearIR contains Loop-specific iterations (`m_has_specific_loops` = true),
 *              loopEnd expressions in the LinearIR may have the same LoopIDs.
 *              Otherwise, when the LinearIR has Solid Loops, loopIDs must be unique!
 * @ingroup snippets
 */

class NormalizeLoopIDs : public Pass {
public:
    OPENVINO_RTTI("NormalizeLoopIDs", "Pass")
    NormalizeLoopIDs(bool has_specific_loops = true) : m_has_specific_loops(has_specific_loops) {}
    bool run(lowered::LinearIR& linear_ir) override;

private:
    bool m_has_specific_loops;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
