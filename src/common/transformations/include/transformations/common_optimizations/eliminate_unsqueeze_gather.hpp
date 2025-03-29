// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EliminateUnsqueezeGather;
class TRANSFORMATIONS_API EliminateGatherUnsqueeze;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Remove Unsqueeze + Gather pair, if Gather gathers data by dimension
 * that was previously added by Unsqueeze
 */

class ov::pass::EliminateUnsqueezeGather : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateUnsqueezeGather");
    EliminateUnsqueezeGather();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Matches Gather ->[Binary Operation]-> Unsqueeze
 * If axis for Gather and Unsqueeze is the same and Gather indices are scalar Unsqueeze is being removed and indices
 * become 1D. Must be executed after SharedOpOptimization -- It is possible to have multiple similar Unsqueeze
 * operations after Gather, so they must be optimized beforehand
 */

class ov::pass::EliminateGatherUnsqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateGatherUnsqueeze");
    EliminateGatherUnsqueeze();
};
