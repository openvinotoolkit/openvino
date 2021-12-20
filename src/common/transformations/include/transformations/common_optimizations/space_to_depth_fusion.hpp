// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SpaceToDepthFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SpaceToDepthFusion transformation replaces following graph:
 * 
 * (any_input) -> StridedSlice -> StridedSlice -> concat
 *          +---> StridedSlice -> StridedSlice ----+
 *          +---> StridedSlice -> StridedSlice ----+
 *          +---> StridedSlice -> StridedSlice ----+
 * 
 * with SpaceToDepth when applicable.
 * 
 */

class ngraph::pass::SpaceToDepthFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SpaceToDepthFusion();
};
