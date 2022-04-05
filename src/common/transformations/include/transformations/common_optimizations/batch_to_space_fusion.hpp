// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API BatchToSpaceFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief BatchToSpaceFusion transformation replaces following graph:
 * Transpose (or Reshape) -> DepthToSpace -> StridedSlice -> Transpose (or Reshape)
 * to BatchToSpace
 * Restrictions:
 * - input rank must be 4
 * - Transpose permutation must be [1, 0, 2, 3]
 * - DepthToSpaceMode must be BLOCKS_FIRST
 */

class ngraph::pass::BatchToSpaceFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("BatchToSpaceFusion", "0");
    BatchToSpaceFusion();
};
