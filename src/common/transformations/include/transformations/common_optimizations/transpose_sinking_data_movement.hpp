// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingDataMovementForward;
class TRANSFORMATIONS_API TransposeSinkingDataMovementBackward;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingDataMovementForward transformation sinks Transpose through BatchToSpace, SpaceToBatch
 * and Pad operations in the forward direction.
 * These operations are categorized as "DataMovement" and are handled in a similar way in this transformation.
 */
class ov::pass::TransposeSinkingDataMovementForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingDataMovementForward", "0");
    TransposeSinkingDataMovementForward();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingDataMovementBackward transformation sinks Transpose through BatchToSpace, SpaceToBatch
 * and Pad operations in the backward direction.
 * These operations are categorized as "DataMovement" and are handled in a similar way in this transformation.
 */
class ov::pass::TransposeSinkingDataMovementBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingDataMovementBackward", "0");
    TransposeSinkingDataMovementBackward();
};
