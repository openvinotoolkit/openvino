// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API PullReshapeThroughDequantization;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief PullReshapeThroughDequantization propagates dequantization operations through Reshape operations.
 * The transformation is used on constant subgraph on weights to prepare a model for next low precision transformations.
 */
class ngraph::pass::low_precision::PullReshapeThroughDequantization : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PullReshapeThroughDequantization(const std::vector<ngraph::element::Type>& inputPrecisions = {});
};
