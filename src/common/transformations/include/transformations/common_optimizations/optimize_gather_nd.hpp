// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API OptimizerGatherND;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief DivideFusion transformation replaces a sub-graph
 * Pow(y, -1) * x or x * Pow(y, -1) with Divide(x,y)
 */
class ov::pass::OptimizerGatherND : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("OptimizerGatherND", "0");
    OptimizerGatherND();
};
