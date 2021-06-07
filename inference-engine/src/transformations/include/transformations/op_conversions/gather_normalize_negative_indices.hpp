// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API GatherNegativeConstIndicesNormalize;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief GatherNegativeConstIndicesNormalize checks if indices value is negative scalar and
 * normalizes it using ShapeOf->Add->Cast subgraph.
 * We need to remove this transformation after adding support of negative indices in
 * future version of Gather operation.
 */
class ngraph::pass::GatherNegativeConstIndicesNormalize : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GatherNegativeConstIndicesNormalize();
};
