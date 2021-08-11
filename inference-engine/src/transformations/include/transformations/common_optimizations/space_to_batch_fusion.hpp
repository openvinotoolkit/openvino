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

class TRANSFORMATIONS_API SpaceToBatchFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief SpaceToBatchFusion transformation replaces following graph:
 * Transpose (or Reshape) -> Pad -> SpaceToDepth -> Transpose (or Reshape)
 * to SpaceToBatch
 * Restrictions:
 * - input rank must be 4
 * - Transpose permutation must be [1, 0, 2, 3]
 * - pad value is 0, PadMode is CONSTANT
 * - SpaceToDepthMode must be BLOCKS_FIRST
 */

class ov::pass::SpaceToBatchFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SpaceToBatchFusion();
};
