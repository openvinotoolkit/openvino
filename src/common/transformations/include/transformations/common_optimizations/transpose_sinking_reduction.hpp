// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingReductionForward;
class TRANSFORMATIONS_API TransposeSinkingReductionBackward;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeReductionForward transformation sinks Transpose through Reduce, Squeeze, Unsqueeze operations
 * in the forward direction.
 */
class ov::pass::TransposeSinkingReductionForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingReductionForward", "0");
    TransposeSinkingReductionForward();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeReductionBackward transformation sinks Transpose through Reduce, Squeeze, Unsqueeze operations
 * in the backward direction.
 */
class ov::pass::TransposeSinkingReductionBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TransposeSinkingReductionBackward", "0");
    TransposeSinkingReductionBackward();
};