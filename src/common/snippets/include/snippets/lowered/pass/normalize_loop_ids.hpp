// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface NormalizeLoopIDs
 * @brief Sort loop IDs in the order of LoopEnd expressions execution and set them evenly: without missed numbers in
 * IDs. Loops might have an arbitrary IDs ordering as a result of optimizations: IDs can be unsorted or non-consecutive
 * (some are missing).
 *        For example,
 *             [Loop -> ID -> new ID]
 *              Loop0 -> 3 -> 0
 *              Loop1 -> 0 -> 1
 *              Loop2 -> 1 -> 2
 *        Note: If the LinearIR contains Loop-specific iterations (`m_has_specific_loops` = true),
 *              loopEnd expressions in the LinearIR may have the same LoopIDs.
 *              Otherwise, when the LinearIR has unified Loops, loopIDs must be unique!
 * @ingroup snippets
 */

class NormalizeLoopIDs : public Pass {
public:
    OPENVINO_RTTI("NormalizeLoopIDs", "", Pass);
    explicit NormalizeLoopIDs(bool has_specific_loops = true) : m_has_specific_loops(has_specific_loops) {}
    bool run(lowered::LinearIR& linear_ir) override;

private:
    // [ original Loop ID -> new normalized and sorted ]
    using IDMapper = std::map<size_t, size_t>;

    static void update_linear_ir(lowered::LinearIR& linear_ir, const IDMapper& loop_id_map);

    bool m_has_specific_loops = true;
};

}  // namespace ov::snippets::lowered::pass
