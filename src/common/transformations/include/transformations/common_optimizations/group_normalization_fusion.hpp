// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GroupNormalizationFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief GroupNormalizationFusion transformation replaces
 * following pattern with fused GroupNormalization op:
 * group_norm_gamma * (instance_norm_gamma * MVN(x) + instance_norm_beta) + group_norm_beta
 * note that instance norm related parameters are optional:
 * - instance_norm_gamma is assumed to be filled with ones if not present in the graph
 * - instance_norm_beta is assumed to be filled with zeros if not present in the graph
 */

class ov::pass::GroupNormalizationFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GroupNormalizationFusion", "0");
    GroupNormalizationFusion();
};
