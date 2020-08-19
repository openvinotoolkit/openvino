// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API NormalizeL2Fusion;
class TRANSFORMATIONS_API NormalizeL2FusionWithMax;
class TRANSFORMATIONS_API NormalizeL2FusionWithAdd;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief NormalizeL2Fusion transformation replaces group of
 * operations: 
 * out[i0, i1, ..., iN] = x[i0, i1, ..., iN] / sqrt(eps_mode(sum(x[j0, ..., jN]**2), eps)
 */
class ngraph::pass::NormalizeL2Fusion: public ngraph::pass::GraphRewrite {
public:
    NormalizeL2Fusion() {
        add_matcher<ngraph::pass::NormalizeL2FusionWithMax>();
        add_matcher<ngraph::pass::NormalizeL2FusionWithAdd>();
    }
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6))) / 6 with a HSwish op.
 */
 class ngraph::pass::NormalizeL2FusionWithMax: public ngraph::pass::MatcherPass {
public:
    NormalizeL2FusionWithMax();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief HSwishFusion transformation replaces a sub-graph (x * (min(Relu(x + 3), 6))) / 6 with a HSwish op.
 */
 class ngraph::pass::NormalizeL2FusionWithAdd: public ngraph::pass::MatcherPass {
public:
    NormalizeL2FusionWithAdd();
};
