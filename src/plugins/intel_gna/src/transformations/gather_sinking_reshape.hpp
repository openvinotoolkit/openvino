// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/** @brief
 *  Moves Gather through Reshape node in a backward sinking propagation
 *  (from the end to the start of the graph).
 *
 *        Any #1        Any #1
 *          |              |
 *       Reshape         Gather
 *          |      =>      |
 *       Gather          Reshape
 *          |              |
 *        Any #2         Any #2
 *
 * - Reshape is an unflatten type operation (one last dimension is expanded into multiple dimensions)
 * - Gather is available for sinking (NoGatherSinkingAttr is not set)
 *
 * This transformation is called from GatherSinkingGeneral.
 */

class GatherSinkingReshapeBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingReshapeBackward", "0");
    GatherSinkingReshapeBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
