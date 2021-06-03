// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API GatherNegativeIndicesNormalize;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief GatherNegativeIndicesNormalize check if indices value is negative and
 * normalize it using ShapeOf->Add->Cast subgraph.
 * We need to remove this transformation after support a negative indices in
 * future version of Gather operation.
 */
class ngraph::pass::GatherNegativeIndicesNormalize : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GatherNegativeIndicesNormalize();
};
