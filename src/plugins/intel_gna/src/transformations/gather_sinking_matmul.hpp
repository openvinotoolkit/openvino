// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/** @brief Gather node through the MatMul operation within forward sinking step
 * in a direction from start to end of the graph.
 *
 *      Any #1 Any #2           Any #1 Any #2
 *       |       |                  |    |
 *      Gather   |        =>        |   reversed Gather
 *           |   |                  |    |
 *          MatMul                  MatMul
 *            |                        |
 *           Any #3                  Any #3
 *
 * or
 *
 *      Any #1 Any #2           Any #1         Any #2
 *       |       |                 |             |
 *       |     Gather    =>  reversed Gather     |
 *       |      |                  |             |
 *        MatMul                         MatMul
 *          |                             |
 *        Any #3                       Any #3
 *
 * This transformation is called called from GatherSinkingGeneral.
 */

class GatherSinkingMatmulForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingMatmulForward", "0");
    GatherSinkingMatmulForward();
};

/** @brief Gather node through the MatMul operation within backward sinking step
 * in a direction from end to start of the graph.
 *
 *      Any #1 Any #2           Any #1    Any #2
 *       |       |                  |       |
 *       |       |   =>           Gather    |
 *       |       |                  |       |
 *        MatMul                      MatMul
 *          |                            |
 *        Gather                       Any3
 *          |
 *        Any #3
 *
 *  or
 *
 *      Any #1 Any #2           Any #1    Any #2
 *       |       |                  |       |
 *       |       |   =>             |      Gather
 *       |       |                  |       |
 *        MatMul                      MatMul
 *          |                            |
 *        Gather                       Any3
 *          |
 *        Any #3
 *
 * Input index depends on the shapes and Gather axis.
 * This transformation is called called from GatherSinkingGeneral.
 */

class GatherSinkingMatmulBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingMatmulBackward", "0");
    GatherSinkingMatmulBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
