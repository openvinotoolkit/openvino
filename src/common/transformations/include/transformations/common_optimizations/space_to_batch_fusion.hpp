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

class TRANSFORMATIONS_API SpaceToBatchFusion;

}  // namespace pass
}  // namespace ngraph

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

class ngraph::pass::SpaceToBatchFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SpaceToBatchFusion", "0");
    SpaceToBatchFusion();
};
