// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class NGRAPH_API Proposal1Scales;
class NGRAPH_API Proposal4Scales;

}  // namespace pass
}  // namespace ngraph

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

class ngraph::pass::Proposal1Scales : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("Proposal1Scales", "0");
    Proposal1Scales();
};

class ngraph::pass::Proposal4Scales : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("Proposal4Scales", "0");
    Proposal4Scales();
};
