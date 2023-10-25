// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief
 * Sinks Transpose through Reshape from the start to the end of the graph.
 *
 *      Any1          Any1
 *       |             |
 *   Transpose       Reshape
 *       |     =>      |
 *    Reshape        Gather
 *       |             |
 *      Any2         Any2
 *
 * Reshape must be tail-flatten: it should squash last mulitple dimensions into one.
 * i.e. [1, 2, 4] => [1, 8]
 *
 * Transpose restrictions:
 * - permute only dims that are flattened by Reshape
 * - 2D permutations
 *   i.e. [0, 2, 1] or [0, 3, 1, 2]
 */
class GatherSinkingTransposeReshapeForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingTransposeReshapeForward", "0");
    GatherSinkingTransposeReshapeForward();
};

/**
 * @brief
 * Sinks Transpose through Reshape from the end to the start of the graph.
 *
 *      Any1          Any1
 *       |             |
 *    Reshape        Gather
 *       |     =>      |
 *    Transpose     Reshape
 *       |             |
 *      Any2         Any2
 *
 * Reshape must be tail-unflatten: it should unsquash last dimension into multiple ones.
 * i.e. [1, 8] => [1, 2, 4]
 *
 * Transpose restrictions:
 * - permute only dims that are unflattened by Reshape
 * - 2D permutations
 *   i.e. [0, 2, 1] or [0, 3, 1, 2]
 */
class GatherSinkingTransposeReshapeBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingTransposeReshapeBackward", "0");
    GatherSinkingTransposeReshapeBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
