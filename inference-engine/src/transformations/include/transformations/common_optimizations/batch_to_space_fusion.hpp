// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BatchToSpaceFusion;

}  // namespace pass
}  // namespace ov

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

class ov::pass::BatchToSpaceFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BatchToSpaceFusion();
};
