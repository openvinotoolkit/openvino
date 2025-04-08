// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API Proposal1Scales;
class TRANSFORMATIONS_API Proposal4Scales;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ProposalScales transformation helps to silently avoid reshape issues on the scale-input of Proposal layer.
 *
 * Expected sub-graph looks like:
 *      Parameter [batch, 3 or 4] -> Reshape [-1] -(in: 3)-> PriorBox
 *
 * PriorBox operation accepts 3 or 4 values as scales from specification standpoint
 * PriorBox uses first set (batch) of scale values to proceed in the plugins
 * According to this we explicitly take first batch of scales with StridedSlice operation
 *
 * Resulting sub-graph:
 *      Parameter [batch, 3 or 4] -> Reshape [-1] -> StridedSlice[0: 3 or 4] -(in: 3)-> PriorBox
 */

class ov::pass::Proposal1Scales : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Proposal1Scales");
    Proposal1Scales();
};

class ov::pass::Proposal4Scales : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("Proposal4Scales");
    Proposal4Scales();
};
