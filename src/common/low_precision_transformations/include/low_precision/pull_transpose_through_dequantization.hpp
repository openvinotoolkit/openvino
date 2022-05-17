// Copyright (C) 2018-2022 Intel Corporation
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

class LP_TRANSFORMATIONS_API PullTransposeThroughDequantization;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief PullTransposeThroughDequantization propagates dequantization operations through Transpose operations.
 * The transformation is used on constant subgraph weights to prepare a model for the next low precision transformations.
 *
 * For more details about the transformation, refer to
 * [PullTransposeThroughDequantization](@ref openvino_docs_OV_UG_lpt_PullTransposeThroughDequantization) page
 * in the Inference Engine Developer Guide.
 */
class ngraph::pass::low_precision::PullTransposeThroughDequantization : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PullTransposeThroughDequantization(const std::vector<ngraph::element::Type>& inputPrecisions = {});
};
