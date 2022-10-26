// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API RandomUniformFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief RandomUniformFusion transformation replaces RandomUniform -> Add or
 * RandomUniform -> Mul subgraph with a RandomUniform and replaces min and max const
 * with corrected values.
 */
class ngraph::pass::RandomUniformFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RandomUniformFusion", "0");
    RandomUniformFusion();
};
