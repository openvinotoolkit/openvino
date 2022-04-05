// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API DivideFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief DivideFusion transformation replaces a sub-graph
 * Pow(y, -1) * x or x * Pow(y, -1) with Divide(x,y)
 */
class ngraph::pass::DivideFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("DivideFusion", "0");
    DivideFusion();
};
