// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SubtractFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SubtractFusion transformation replaces a sub-graph
 * Mul(y, -1) + x or x + Mul(y, -1) with Subtract(x,y)
 */
class ov::pass::SubtractFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SubtractFusion", "0");
    SubtractFusion();
};
