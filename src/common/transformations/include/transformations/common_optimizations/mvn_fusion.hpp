// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "transformations_visibility.hpp"

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
    OPENVINO_RTTI("MVNFusionWithoutConstants", "0");
    MVNFusionWithoutConstants();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief MVNFusion transformation replaces group of
 * operations: gamma * (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) + beta to MVN
 * op.
 */
class ov::pass::MVNFusionWithConstantsInside : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MVNFusionWithConstantsInside", "0");
    MVNFusionWithConstantsInside();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief MVNFusion transformation replaces various sub-graphs with a MVN op.
 */
class ov::pass::MVNFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("MVNFusion", "0");
    MVNFusion() {
        add_matcher<ov::pass::MVNFusionWithoutConstants>();
        add_matcher<ov::pass::MVNFusionWithConstantsInside>();
    }
};
