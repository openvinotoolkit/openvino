// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingConcatForward;
class TRANSFORMATIONS_API TransposeSinkingConcatBackward;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingConcatForward transformation sinks Transpose through Concat operation
 * in the forward direction.
 */
class ov::pass::TransposeSinkingConcatForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingConcatForward", "0");
    TransposeSinkingConcatForward();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingConcatBackward transformation sinks Transpose through Concat operation
 * in the backward direction.
 */
class ov::pass::TransposeSinkingConcatBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingConcatBackward", "0");
    TransposeSinkingConcatBackward();
};
