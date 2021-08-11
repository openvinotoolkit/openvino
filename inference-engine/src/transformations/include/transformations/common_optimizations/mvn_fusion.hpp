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

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MVNFusion;
class TRANSFORMATIONS_API MVNFusionWithoutConstants;
class TRANSFORMATIONS_API MVNFusionWithConstantsInside;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief MVNFusion transformation replaces group of
 * operations: (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) to MVN op.
 */
class ov::pass::MVNFusionWithoutConstants : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MVNFusionWithoutConstants();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief MVNFusion transformation replaces group of
 * operations: gamma * (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) + beta to MVN op.
 */
class ov::pass::MVNFusionWithConstantsInside : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MVNFusionWithConstantsInside();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief MVNFusion transformation replaces various sub-graphs with a MVN op.
 */
class ov::pass::MVNFusion: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    MVNFusion() {
        add_matcher<ov::pass::MVNFusionWithoutConstants>();
        add_matcher<ov::pass::MVNFusionWithConstantsInside>();
    }
};
