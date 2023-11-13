// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/** @brief
 * Moves Gather through Split layer in a backward step propagation.
 * This transformation is called from GatherSinkingGeneral.
 *
 * Converts subgraph
 *                Any #1
 *                  |
 *                Split
 *    |             |          |
 *   Any #2 ...    Gather ... Any #N
 *                  |
 *                Any #K
 * to
 *                Any #1
 *                  |
 *               Gather
 *                  |
 *                Split
 *    |             |          |
 *   Any #2 ...    Any #K ... Any #N
 *
 *
 */

class GatherSinkingSplitBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingBinaryBackward", "0");
    GatherSinkingSplitBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
