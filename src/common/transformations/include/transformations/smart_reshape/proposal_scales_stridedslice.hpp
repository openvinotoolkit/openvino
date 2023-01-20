// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <vector>

#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API Proposal1Scales;
class TRANSFORMATIONS_API Proposal4Scales;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
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
    OPENVINO_RTTI("Proposal1Scales", "0");
    Proposal1Scales();
};

class ov::pass::Proposal4Scales : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("Proposal4Scales", "0");
    Proposal4Scales();
};

namespace ngraph {
namespace pass {
using ov::pass::Proposal1Scales;
using ov::pass::Proposal4Scales;
}  // namespace pass
}  // namespace ngraph
