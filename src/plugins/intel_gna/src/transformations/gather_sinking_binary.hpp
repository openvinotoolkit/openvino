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

/**
 * @brief
 * Moves Gather layer forward from the start to the end of the graph
 * through the BinaryElementwiseArithmetic operations (it is called from GatherSinkingGeneral transformation).
 * Gather layer is moved from the Binary input to the Binary output. Reversed Gather layer is moved
 * to another Binary input.
 * Reversed Gather is a layer which, when put after the original gather, results in a subgraph that has no effect
 * Reversed Gather layer is expected to be pushed by backward GatherSinking to another
 * Model input or a constant. After all sinking operations we hope to find all Gather
 * layers on Model inputs and execute them on CPU or before constants and fold them.
 * Transformation restrictions:
 * - Gather has only 1D indices
 * - all nodes have static ranks
 *
 *    Any1                 Any1   Any2
 *     |                      |       |
 *    Gather Any2             |  Reversed-Gather
 *      |    |        =>      |       |
 *      Binary                 Binary
 *        |                       |
 *      Any3                   Gather
 *                                |
 *                              Any3
 *
 *           Any1            Any2      Any1
 *            |               |         |
 *    Any2 Gather      Reversed-Gather  |
 *      |    |        =>      |         |
 *      Binary                  Binary
 *        |                       |
 *      Any3                   Gather
 *                                |
 *                              Any3
 *
 * All GatherSinking tranformations are designed to work in 2 steps:
 * - forward push
 * - backward push
 * Add flag into Gather layer rt_info that prevents backward sinking if the next layer
 * after Gather does not support by GatherSinking transformations. That is done to
 * prevent backward pushing the layer that already pushed forward through the graph.
 */
class GatherSinkingBinaryForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingBinaryForward", "0");
    GatherSinkingBinaryForward();
};

/**
 * @brief
 * Moves Gather layer backward from the end to the start of the graph
 *
 *    Any1  Any2     Any1   Any2
 *     |     |        |     |
 *     Binary   =>  Gather Gather
 *        |            |    |
 *      Gather         Binary
 *        |              |
 *       Any3           Any3
 *
 *    Any1  Any2     Any1     Any2
 *     |     |         |       |
 *     Binary  =>    Gather Gather
 *     |    |          |     |
 *   Gather Gather      Binary
 *     |     |          |   |
 *    Any3  Any4       Any3 Any4
 *
 * Moves Gather layer backward only if:
 * - Gather is not marked as non-sinkable
 * - all Binary consumers are Gather layers
 * - all that Gather layers equal each other
 * - all Gather layers have only 1D indices
 * - all nodes have static ranks
 *
 * This transformation is called called from GatherSinkingGeneral.
 */
class GatherSinkingBinaryBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingBinaryBackward", "0");
    GatherSinkingBinaryBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov