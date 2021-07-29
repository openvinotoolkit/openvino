// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API MVNFusion;
class TRANSFORMATIONS_API MVNFusionWithoutConstants;
class TRANSFORMATIONS_API MVNFusionWithConstantsInside;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MVNFusion transformation replaces group of
 * operations: (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) to MVN op.
 */
class ngraph::pass::MVNFusionWithoutConstants : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MVNFusionWithoutConstants();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief MVNFusion transformation replaces group of
 * operations: gamma * (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) + beta to MVN op.
 */
class ngraph::pass::MVNFusionWithConstantsInside : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MVNFusionWithConstantsInside();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief MVNFusion transformation replaces various sub-graphs with a MVN op.
 */
class ngraph::pass::MVNFusion: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    MVNFusion() {
        add_matcher<ngraph::pass::MVNFusionWithoutConstants>();
        add_matcher<ngraph::pass::MVNFusionWithConstantsInside>();
    }
};
