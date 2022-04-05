// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SubtractFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SubtractFusion transformation replaces a sub-graph
 * Mul(y, -1) + x or x + Mul(y, -1) with Subtract(x,y)
 */
class ngraph::pass::SubtractFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SubtractFusion", "0");
    SubtractFusion();
};
