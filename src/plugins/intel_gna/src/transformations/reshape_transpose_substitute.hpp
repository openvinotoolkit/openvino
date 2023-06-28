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
 * @brief Substitute Reshape with Transpose if it possible.
 * It is usefull with Tranpose/Gather sinking transformations to simplify Transpose movement.
 * TransposeSinking is not possible to use with Reshapes (except some cases when Reshape is Squeeze/Unsqueeze).
 * GatherSinking is much harder to implement for all existed layer types.
 *
 * any layer          any layer
 *    |                  |
 *  Reshape   =>      Transpose
 *    |                  |
 * any layer          any layer
 */
class ReshapeTransposeSubstitute : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeTransposeSubstitute", "0");
    ReshapeTransposeSubstitute();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
